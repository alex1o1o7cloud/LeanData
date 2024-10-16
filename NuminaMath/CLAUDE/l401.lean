import Mathlib

namespace NUMINAMATH_CALUDE_alternate_tree_planting_l401_40158

/-- The number of ways to arrange n items from a set of m items, where order matters -/
def arrangements (m n : ℕ) : ℕ := sorry

/-- The number of ways to plant w willow trees and p poplar trees alternately in a row -/
def alternate_tree_arrangements (w p : ℕ) : ℕ :=
  2 * arrangements w w * arrangements p p

theorem alternate_tree_planting :
  alternate_tree_arrangements 4 4 = 1152 := by sorry

end NUMINAMATH_CALUDE_alternate_tree_planting_l401_40158


namespace NUMINAMATH_CALUDE_book_cost_in_cny_l401_40123

/-- Exchange rate from US dollar to Namibian dollar -/
def usd_to_nad : ℚ := 7

/-- Exchange rate from US dollar to Chinese yuan -/
def usd_to_cny : ℚ := 6

/-- Cost of the book in Namibian dollars -/
def book_cost_nad : ℚ := 168

/-- Calculate the cost of the book in Chinese yuan -/
def book_cost_cny : ℚ := book_cost_nad * (usd_to_cny / usd_to_nad)

/-- Theorem stating that the book costs 144 Chinese yuan -/
theorem book_cost_in_cny : book_cost_cny = 144 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_in_cny_l401_40123


namespace NUMINAMATH_CALUDE_sandy_current_fingernail_length_l401_40154

/-- Sandy's current age in years -/
def current_age : ℕ := 12

/-- Sandy's age when she achieves the world record in years -/
def record_age : ℕ := 32

/-- The world record for longest fingernails in inches -/
def world_record : ℝ := 26

/-- Sandy's fingernail growth rate in inches per month -/
def growth_rate : ℝ := 0.1

/-- The number of months in a year -/
def months_per_year : ℕ := 12

theorem sandy_current_fingernail_length :
  world_record - (growth_rate * months_per_year * (record_age - current_age : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandy_current_fingernail_length_l401_40154


namespace NUMINAMATH_CALUDE_polar_equivalence_l401_40198

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Checks if two polar points are equivalent -/
def polar_equivalent (p1 p2 : PolarPoint) : Prop :=
  p1.r * (Real.cos p1.θ) = p2.r * (Real.cos p2.θ) ∧
  p1.r * (Real.sin p1.θ) = p2.r * (Real.sin p2.θ)

theorem polar_equivalence :
  let p1 : PolarPoint := ⟨6, 4*Real.pi/3⟩
  let p2 : PolarPoint := ⟨-6, Real.pi/3⟩
  polar_equivalent p1 p2 := by
  sorry

end NUMINAMATH_CALUDE_polar_equivalence_l401_40198


namespace NUMINAMATH_CALUDE_manny_cookie_pies_l401_40115

theorem manny_cookie_pies :
  ∀ (num_pies : ℕ) (num_classmates : ℕ) (num_teacher : ℕ) (slices_per_pie : ℕ) (slices_left : ℕ),
    num_classmates = 24 →
    num_teacher = 1 →
    slices_per_pie = 10 →
    slices_left = 4 →
    (num_pies * slices_per_pie = num_classmates + num_teacher + 1 + slices_left) →
    num_pies = 3 :=
by
  sorry

#check manny_cookie_pies

end NUMINAMATH_CALUDE_manny_cookie_pies_l401_40115


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_350_l401_40132

def closest_perfect_square (n : ℕ) : ℕ :=
  let root := n.sqrt
  if (root + 1)^2 - n < n - root^2
  then (root + 1)^2
  else root^2

theorem closest_perfect_square_to_350 :
  closest_perfect_square 350 = 361 := by
  sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_350_l401_40132


namespace NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l401_40109

/-- An arithmetic progression with a term equal to zero -/
theorem arithmetic_progression_zero_term
  (a : ℕ → ℝ)  -- The arithmetic progression
  (n m : ℕ)    -- Indices of the given terms
  (h : a (2 * n) / a (2 * m) = -1)  -- Given condition
  : ∃ k, a k = 0 ∧ k = n + m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_zero_term_l401_40109


namespace NUMINAMATH_CALUDE_prime_factor_sum_l401_40162

theorem prime_factor_sum (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 11^z = 825) : 
  w + 2*x + 3*y + 4*z = 12 := by
sorry

end NUMINAMATH_CALUDE_prime_factor_sum_l401_40162


namespace NUMINAMATH_CALUDE_japanese_students_count_l401_40145

theorem japanese_students_count (chinese : ℕ) (korean : ℕ) (japanese : ℕ) 
  (h1 : korean = (6 * chinese) / 11)
  (h2 : japanese = chinese / 8)
  (h3 : korean = 48) : 
  japanese = 11 := by
sorry

end NUMINAMATH_CALUDE_japanese_students_count_l401_40145


namespace NUMINAMATH_CALUDE_complex_equation_l401_40152

/-- Given x ∈ ℝ, y is a pure imaginary number, and (x-y)i = 2-i, then x+y = -1+2i -/
theorem complex_equation (x : ℝ) (y : ℂ) (h1 : y.re = 0) (h2 : (x - y) * Complex.I = 2 - Complex.I) : 
  x + y = -1 + 2 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_equation_l401_40152


namespace NUMINAMATH_CALUDE_abs_neg_abs_square_minus_one_eq_zero_l401_40134

theorem abs_neg_abs_square_minus_one_eq_zero :
  |(-|(-1 + 2)|)^2 - 1| = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_abs_square_minus_one_eq_zero_l401_40134


namespace NUMINAMATH_CALUDE_remove_seven_improves_mean_l401_40165

def scores : List ℕ := [6, 7, 7, 8, 8, 8, 9, 10]

def mode (l : List ℕ) : ℕ := sorry

def range (l : List ℕ) : ℕ := sorry

def mean (l : List ℕ) : ℚ := sorry

def remove_score (s : List ℕ) (n : ℕ) : List ℕ := sorry

theorem remove_seven_improves_mean :
  let original_scores := scores
  let new_scores := remove_score original_scores 7
  mode new_scores = mode original_scores ∧
  range new_scores = range original_scores ∧
  mean new_scores > mean original_scores :=
sorry

end NUMINAMATH_CALUDE_remove_seven_improves_mean_l401_40165


namespace NUMINAMATH_CALUDE_negation_of_proposition_l401_40184

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 + 3 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 3 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l401_40184


namespace NUMINAMATH_CALUDE_expression_evaluation_l401_40126

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := -1
  5 * x^2 - 2 * (3 * y^2 + 6 * x) + (2 * y^2 - 5 * x^2) = 8 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l401_40126


namespace NUMINAMATH_CALUDE_circle_properties_l401_40199

-- Define the circle's circumference
def circumference : ℝ := 36

-- Theorem statement
theorem circle_properties :
  let radius := circumference / (2 * Real.pi)
  let diameter := 2 * radius
  let area := Real.pi * radius^2
  (radius = 18 / Real.pi) ∧
  (diameter = 36 / Real.pi) ∧
  (area = 324 / Real.pi) := by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l401_40199


namespace NUMINAMATH_CALUDE_milk_problem_l401_40131

theorem milk_problem (initial_milk : ℚ) (tim_fraction : ℚ) (kim_fraction : ℚ) : 
  initial_milk = 3/4 →
  tim_fraction = 1/3 →
  kim_fraction = 1/2 →
  kim_fraction * (initial_milk - tim_fraction * initial_milk) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l401_40131


namespace NUMINAMATH_CALUDE_geometric_progression_naturals_l401_40105

theorem geometric_progression_naturals (a₁ : ℕ) (q : ℚ) :
  (∃ (a₁₀ a₃₀ : ℕ), a₁₀ = a₁ * q^9 ∧ a₃₀ = a₁ * q^29) →
  ∃ (a₂₀ : ℕ), a₂₀ = a₁ * q^19 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_naturals_l401_40105


namespace NUMINAMATH_CALUDE_a_values_l401_40163

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l401_40163


namespace NUMINAMATH_CALUDE_remainder_sum_l401_40178

theorem remainder_sum (n : ℤ) : n % 20 = 11 → (n % 4 + n % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l401_40178


namespace NUMINAMATH_CALUDE_third_set_candy_count_l401_40164

/-- Represents the number of candies of a specific type in a set -/
structure CandyCount where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- Represents the total candy distribution across three sets -/
structure CandyDistribution where
  set1 : CandyCount
  set2 : CandyCount
  set3 : CandyCount

/-- The conditions of the candy distribution problem -/
def validDistribution (d : CandyDistribution) : Prop :=
  -- Total number of each type is equal across all sets
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.chocolate + d.set2.chocolate + d.set3.chocolate ∧
  d.set1.hard + d.set2.hard + d.set3.hard = 
  d.set1.gummy + d.set2.gummy + d.set3.gummy ∧
  -- First set conditions
  d.set1.chocolate = d.set1.gummy ∧
  d.set1.hard = d.set1.chocolate + 7 ∧
  -- Second set conditions
  d.set2.hard = d.set2.chocolate ∧
  d.set2.gummy = d.set2.hard - 15 ∧
  -- Third set condition
  d.set3.hard = 0

/-- The main theorem stating that any valid distribution has 29 candies in the third set -/
theorem third_set_candy_count (d : CandyDistribution) : 
  validDistribution d → d.set3.chocolate + d.set3.gummy = 29 := by
  sorry


end NUMINAMATH_CALUDE_third_set_candy_count_l401_40164


namespace NUMINAMATH_CALUDE_z_in_terms_of_x_l401_40183

theorem z_in_terms_of_x (p : ℝ) (x z : ℝ) 
  (hx : x = 2 + 3^p) 
  (hz : z = 2 + 3^(-p)) : 
  z = (2*x - 3) / (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_z_in_terms_of_x_l401_40183


namespace NUMINAMATH_CALUDE_point_on_y_axis_l401_40104

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define what it means for a point to be on the y-axis
def onYAxis (p : Point2D) : Prop := p.x = 0

-- Theorem statement
theorem point_on_y_axis (p : Point2D) : onYAxis p ↔ p.x = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_l401_40104


namespace NUMINAMATH_CALUDE_technician_permanent_percentage_l401_40119

def factory_workforce (total_workers : ℝ) : Prop :=
  let technicians := 0.8 * total_workers
  let non_technicians := 0.2 * total_workers
  let permanent_non_technicians := 0.2 * non_technicians
  let temporary_workers := 0.68 * total_workers
  ∃ (permanent_technicians : ℝ),
    permanent_technicians + permanent_non_technicians = total_workers - temporary_workers ∧
    permanent_technicians / technicians = 0.35

theorem technician_permanent_percentage :
  ∀ (total_workers : ℝ), total_workers > 0 → factory_workforce total_workers :=
by sorry

end NUMINAMATH_CALUDE_technician_permanent_percentage_l401_40119


namespace NUMINAMATH_CALUDE_total_amount_calculation_l401_40155

/-- The total amount Kanul had -/
def T : ℝ := sorry

/-- Theorem stating the relationship between the total amount and the expenses -/
theorem total_amount_calculation :
  T = 3000 + 2000 + 0.1 * T ∧ T = 5000 / 0.9 := by sorry

end NUMINAMATH_CALUDE_total_amount_calculation_l401_40155


namespace NUMINAMATH_CALUDE_negation_equivalence_l401_40172

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x₀ : ℝ, |x₀ - 2| + |x₀ - 4| ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l401_40172


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l401_40185

theorem exponential_equation_solution :
  ∃ x : ℝ, (3 : ℝ) ^ (x - 2) = 9 ^ (x + 1) ∧ x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l401_40185


namespace NUMINAMATH_CALUDE_total_messages_equals_680_l401_40182

/-- The total number of messages sent by Alina and Lucia over three days -/
def total_messages (lucia_day1 : ℕ) (alina_difference : ℕ) : ℕ :=
  let alina_day1 := lucia_day1 - alina_difference
  let lucia_day2 := lucia_day1 / 3
  let alina_day2 := 2 * alina_day1
  let lucia_day3 := lucia_day1
  let alina_day3 := alina_day1
  lucia_day1 + lucia_day2 + lucia_day3 + alina_day1 + alina_day2 + alina_day3

/-- Theorem stating that the total number of messages sent over three days is 680 -/
theorem total_messages_equals_680 :
  total_messages 120 20 = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_messages_equals_680_l401_40182


namespace NUMINAMATH_CALUDE_dartboard_section_angle_l401_40127

/-- Represents a circular dartboard divided into sections -/
structure Dartboard where
  /-- The probability of a dart landing in a particular section -/
  section_probability : ℝ
  /-- The central angle of the section in degrees -/
  section_angle : ℝ

/-- 
Theorem: For a circular dartboard divided into sections by radius lines, 
if the probability of a dart landing in a particular section is 1/4, 
then the central angle of that section is 90 degrees.
-/
theorem dartboard_section_angle (d : Dartboard) 
  (h_prob : d.section_probability = 1/4) : 
  d.section_angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_section_angle_l401_40127


namespace NUMINAMATH_CALUDE_inequality_proof_l401_40125

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l401_40125


namespace NUMINAMATH_CALUDE_card_selection_count_l401_40186

/-- Represents a card with two sides -/
structure Card where
  red : Nat
  blue : Nat
  red_in_range : red ≥ 1 ∧ red ≤ 12
  blue_in_range : blue ≥ 1 ∧ blue ≤ 12

/-- The set of all possible cards -/
def all_cards : Finset Card :=
  sorry

/-- A card is a duplicate if both sides have the same number -/
def is_duplicate (c : Card) : Prop :=
  c.red = c.blue

/-- Two cards have no common numbers -/
def no_common_numbers (c1 c2 : Card) : Prop :=
  c1.red ≠ c2.red ∧ c1.red ≠ c2.blue ∧ c1.blue ≠ c2.red ∧ c1.blue ≠ c2.blue

/-- The set of valid card pairs -/
def valid_pairs : Finset (Card × Card) :=
  sorry

theorem card_selection_count :
  Finset.card valid_pairs = 1386 :=
sorry

end NUMINAMATH_CALUDE_card_selection_count_l401_40186


namespace NUMINAMATH_CALUDE_three_card_picks_count_l401_40117

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- The number of ways to pick three different cards from a standard deck where order matters -/
def threeCardPicks (d : Deck) : ℕ :=
  52 * 51 * 50

/-- Theorem stating that the number of ways to pick three different cards from a standard 
    52-card deck, where order matters, is equal to 132600 -/
theorem three_card_picks_count (d : Deck) : threeCardPicks d = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_picks_count_l401_40117


namespace NUMINAMATH_CALUDE_triangle_side_length_l401_40106

/-- Given a triangle DEF with side lengths and a median, prove the length of DF. -/
theorem triangle_side_length (DE EF DM : ℝ) (hDE : DE = 7) (hEF : EF = 10) (hDM : DM = 5) :
  ∃ (DF : ℝ), DF = Real.sqrt 149 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l401_40106


namespace NUMINAMATH_CALUDE_platform_length_l401_40107

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

end NUMINAMATH_CALUDE_platform_length_l401_40107


namespace NUMINAMATH_CALUDE_sum_greater_than_three_l401_40129

theorem sum_greater_than_three (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_ineq : a * b + b * c + c * a > a + b + c) : 
  a + b + c > 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_three_l401_40129


namespace NUMINAMATH_CALUDE_correct_employee_count_l401_40116

/-- The number of employees in Kim's office -/
def num_employees : ℕ := 9

/-- The total time Kim spends on her morning routine in minutes -/
def total_time : ℕ := 50

/-- The time Kim spends making coffee in minutes -/
def coffee_time : ℕ := 5

/-- The time Kim spends per employee for status update in minutes -/
def status_update_time : ℕ := 2

/-- The time Kim spends per employee for payroll update in minutes -/
def payroll_update_time : ℕ := 3

/-- Theorem stating that the number of employees is correct given the conditions -/
theorem correct_employee_count :
  num_employees * (status_update_time + payroll_update_time) + coffee_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_correct_employee_count_l401_40116


namespace NUMINAMATH_CALUDE_octal_computation_l401_40122

/-- Converts a decimal number to its octal representation -/
def toOctal (n : ℕ) : ℕ := sorry

/-- Multiplies two octal numbers -/
def octalMultiply (a b : ℕ) : ℕ := sorry

/-- Divides an octal number by another octal number -/
def octalDivide (a b : ℕ) : ℕ := sorry

theorem octal_computation : 
  let a := toOctal 254
  let b := toOctal 170
  let c := toOctal 4
  octalDivide (octalMultiply a b) c = 3156 := by sorry

end NUMINAMATH_CALUDE_octal_computation_l401_40122


namespace NUMINAMATH_CALUDE_triangle_tan_b_l401_40144

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
theorem triangle_tan_b (a b c : ℝ) (A B C : ℝ) :
  /- a², b², c² form an arithmetic sequence -/
  (a^2 + c^2 = 2*b^2) →
  /- Area of triangle ABC is b²/3 -/
  (1/2 * a * c * Real.sin B = b^2/3) →
  /- Law of cosines -/
  (b^2 = a^2 + c^2 - 2*a*c*Real.cos B) →
  /- Then tan B = 4/3 -/
  Real.tan B = 4/3 := by
sorry

end NUMINAMATH_CALUDE_triangle_tan_b_l401_40144


namespace NUMINAMATH_CALUDE_complex_equation_solution_l401_40149

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 3 + 4 * Complex.I) → z = 4 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l401_40149


namespace NUMINAMATH_CALUDE_find_y_l401_40133

theorem find_y (x : ℝ) (y : ℝ) : 
  ((100 + 200 + 300 + x) / 4 = 250) →
  ((300 + 150 + 100 + x + y) / 5 = 200) →
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_l401_40133


namespace NUMINAMATH_CALUDE_chicken_count_l401_40146

/-- The number of chickens in different locations and their relationships --/
theorem chicken_count :
  ∀ (coop run free_range barn : ℕ),
  coop = 14 →
  run = 2 * coop →
  5 * (coop + run) = 2 * free_range →
  2 * barn = coop →
  free_range = 105 := by
  sorry

end NUMINAMATH_CALUDE_chicken_count_l401_40146


namespace NUMINAMATH_CALUDE_certain_number_problem_l401_40100

theorem certain_number_problem : ∃ x : ℝ, 0.85 * x = (4/5 * 25) + 14 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l401_40100


namespace NUMINAMATH_CALUDE_farm_animals_l401_40157

theorem farm_animals (sheep ducks : ℕ) : 
  sheep + ducks = 15 → 
  4 * sheep + 2 * ducks = 22 + 2 * (sheep + ducks) → 
  sheep = 11 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l401_40157


namespace NUMINAMATH_CALUDE_sum_interior_angles_decagon_l401_40173

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A decagon is a polygon with 10 sides -/
def decagon_sides : ℕ := 10

/-- Theorem: The sum of the interior angles of a decagon is 1440 degrees -/
theorem sum_interior_angles_decagon :
  sum_interior_angles decagon_sides = 1440 := by sorry

end NUMINAMATH_CALUDE_sum_interior_angles_decagon_l401_40173


namespace NUMINAMATH_CALUDE_garage_cleaning_trips_l401_40187

theorem garage_cleaning_trips (jean_trips bill_trips total_trips : ℕ) : 
  jean_trips = 23 → 
  jean_trips = bill_trips + 6 → 
  total_trips = jean_trips + bill_trips → 
  total_trips = 40 := by
sorry

end NUMINAMATH_CALUDE_garage_cleaning_trips_l401_40187


namespace NUMINAMATH_CALUDE_isosceles_triangle_23_perimeter_l401_40136

-- Define an isosceles triangle with side lengths 2 and 3
structure IsoscelesTriangle23 where
  base : ℝ
  side : ℝ
  is_isosceles : (base = 2 ∧ side = 3) ∨ (base = 3 ∧ side = 2)

-- Define the perimeter of the triangle
def perimeter (t : IsoscelesTriangle23) : ℝ := t.base + 2 * t.side

-- Theorem statement
theorem isosceles_triangle_23_perimeter :
  ∀ t : IsoscelesTriangle23, perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_23_perimeter_l401_40136


namespace NUMINAMATH_CALUDE_apple_picking_solution_l401_40181

/-- Represents the apple picking problem --/
def apple_picking_problem (total : ℕ) (first_day_fraction : ℚ) (remaining : ℕ) : Prop :=
  let first_day := (total : ℚ) * first_day_fraction
  let second_day := 2 * first_day
  let third_day := (total : ℚ) - remaining - first_day - second_day
  (third_day - first_day) = 20

/-- Theorem stating the solution to the apple picking problem --/
theorem apple_picking_solution :
  apple_picking_problem 200 (1/5) 20 := by
  sorry


end NUMINAMATH_CALUDE_apple_picking_solution_l401_40181


namespace NUMINAMATH_CALUDE_target_number_position_l401_40110

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

end NUMINAMATH_CALUDE_target_number_position_l401_40110


namespace NUMINAMATH_CALUDE_doug_had_22_marbles_l401_40137

/-- Calculates the initial number of marbles Doug had -/
def dougs_initial_marbles (eds_marbles : ℕ) (difference : ℕ) : ℕ :=
  eds_marbles - difference

theorem doug_had_22_marbles (eds_marbles : ℕ) (difference : ℕ) 
  (h1 : eds_marbles = 27) 
  (h2 : difference = 5) : 
  dougs_initial_marbles eds_marbles difference = 22 := by
sorry

end NUMINAMATH_CALUDE_doug_had_22_marbles_l401_40137


namespace NUMINAMATH_CALUDE_geometric_sum_formula_l401_40177

/-- Geometric sequence with first term 1 and common ratio 1/3 -/
def geometric_sequence (n : ℕ) : ℚ :=
  (1 / 3) ^ (n - 1)

/-- Sum of the first n terms of the geometric sequence -/
def geometric_sum (n : ℕ) : ℚ :=
  (3 - geometric_sequence n) / 2

/-- Theorem: The sum of the first n terms of the geometric sequence
    is equal to (3 - a_n) / 2 -/
theorem geometric_sum_formula (n : ℕ) :
  geometric_sum n = (3 - geometric_sequence n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_formula_l401_40177


namespace NUMINAMATH_CALUDE_average_age_decrease_l401_40135

theorem average_age_decrease (original_average : ℝ) (new_students : ℕ) (new_average : ℝ) (original_strength : ℕ) :
  original_average = 40 →
  new_students = 15 →
  new_average = 32 →
  original_strength = 15 →
  let total_students := original_strength + new_students
  let new_total_age := original_average * original_strength + new_average * new_students
  let final_average := new_total_age / total_students
  40 - final_average = 4 :=
by sorry

end NUMINAMATH_CALUDE_average_age_decrease_l401_40135


namespace NUMINAMATH_CALUDE_simplify_expression_1_l401_40128

theorem simplify_expression_1 (x : ℝ) : 
  5*x^2 + x + 3 + 4*x - 8*x^2 - 2 = -3*x^2 + 5*x + 1 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l401_40128


namespace NUMINAMATH_CALUDE_waterfall_flow_rate_l401_40190

/-- The waterfall problem -/
theorem waterfall_flow_rate 
  (basin_capacity : ℝ) 
  (leak_rate : ℝ) 
  (fill_time : ℝ) 
  (h1 : basin_capacity = 260) 
  (h2 : leak_rate = 4) 
  (h3 : fill_time = 13) : 
  ∃ (flow_rate : ℝ), flow_rate = 24 ∧ 
  fill_time * flow_rate - fill_time * leak_rate = basin_capacity :=
sorry

end NUMINAMATH_CALUDE_waterfall_flow_rate_l401_40190


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l401_40150

theorem right_triangle_cosine (a b c : ℝ) (h1 : a = 9) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) :
  (a / c) = (3 : ℝ) / 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l401_40150


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l401_40159

theorem distinct_prime_factors_of_90 : Nat.card (Nat.factors 90).toFinset = 3 := by
  sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_90_l401_40159


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l401_40148

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l401_40148


namespace NUMINAMATH_CALUDE_polygon_congruence_l401_40180

/-- A convex polygon in the plane -/
structure ConvexPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  convex : sorry -- Convexity condition

/-- The side length between two consecutive vertices of a polygon -/
def sideLength (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- The angle at a vertex of a polygon -/
def angle (p : ConvexPolygon n) (i : Fin n) : ℝ := sorry

/-- Two polygons are congruent if there exists a rigid motion that maps one to the other -/
def congruent (p q : ConvexPolygon n) : Prop := sorry

/-- Main theorem: Two convex n-gons with equal corresponding sides and n-3 equal corresponding angles are congruent -/
theorem polygon_congruence (n : ℕ) (p q : ConvexPolygon n) 
  (h_sides : ∀ i : Fin n, sideLength p i = sideLength q i)
  (h_angles : ∃ (s : Finset (Fin n)), s.card = n - 3 ∧ ∀ i ∈ s, angle p i = angle q i) :
  congruent p q :=
sorry

end NUMINAMATH_CALUDE_polygon_congruence_l401_40180


namespace NUMINAMATH_CALUDE_square_cut_perimeter_l401_40156

/-- Given a square with side length 2a and a line y = 2x/3 cutting through it,
    the perimeter of one piece divided by a is 6 + (2√13 + 3√2)/3 -/
theorem square_cut_perimeter (a : ℝ) (a_pos : a > 0) :
  let square := {(x, y) | -a ≤ x ∧ x ≤ a ∧ -a ≤ y ∧ y ≤ a}
  let line := {(x, y) | y = (2/3) * x}
  let piece := {p ∈ square | p.2 ≤ (2/3) * p.1 ∨ (p.1 = a ∧ p.2 ≥ (2/3) * a) ∨ (p.1 = -a ∧ p.2 ≤ -(2/3) * a)}
  let perimeter := Real.sqrt ((2*a)^2 + ((4*a)/3)^2) + (4*a)/3 + 2*a + a * Real.sqrt 2
  perimeter / a = 6 + (2 * Real.sqrt 13 + 3 * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_cut_perimeter_l401_40156


namespace NUMINAMATH_CALUDE_value_of_expression_l401_40120

theorem value_of_expression (a : ℝ) (h : a^2 - 2*a - 2 = 3) : 3*a*(a-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l401_40120


namespace NUMINAMATH_CALUDE_hexagons_in_50th_ring_l401_40176

/-- The number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_ring (n : ℕ) : ℕ := 6 * n

/-- Theorem: The number of hexagons in the 50th ring is 300 -/
theorem hexagons_in_50th_ring : hexagons_in_ring 50 = 300 := by
  sorry

end NUMINAMATH_CALUDE_hexagons_in_50th_ring_l401_40176


namespace NUMINAMATH_CALUDE_at_least_fifteen_equal_differences_l401_40194

theorem at_least_fifteen_equal_differences
  (a : Fin 100 → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bounded : ∀ i, 1 ≤ a i ∧ a i ≤ 400)
  (h_increasing : ∀ i j, i < j → a i < a j) :
  ∃ (v : ℕ) (s : Finset (Fin 99)),
    s.card ≥ 15 ∧ ∀ i ∈ s, a (i + 1) - a i = v :=
sorry

end NUMINAMATH_CALUDE_at_least_fifteen_equal_differences_l401_40194


namespace NUMINAMATH_CALUDE_ammonium_chloride_required_l401_40167

/-- Represents a chemical substance with its coefficient in a chemical equation -/
structure ChemicalTerm where
  coefficient : ℕ
  substance : String

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  reactants : List ChemicalTerm
  products : List ChemicalTerm

/-- Checks if a chemical equation is balanced -/
def isBalanced (equation : ChemicalEquation) : Bool := sorry

/-- The chemical equation for the reaction -/
def reactionEquation : ChemicalEquation := {
  reactants := [
    { coefficient := 1, substance := "NH4Cl" },
    { coefficient := 1, substance := "KOH" }
  ],
  products := [
    { coefficient := 1, substance := "NH3" },
    { coefficient := 1, substance := "H2O" },
    { coefficient := 1, substance := "KCl" }
  ]
}

theorem ammonium_chloride_required (equation : ChemicalEquation) 
  (h1 : equation = reactionEquation) 
  (h2 : isBalanced equation) :
  ∃ (nh4cl : ChemicalTerm), 
    nh4cl ∈ equation.reactants ∧ 
    nh4cl.substance = "NH4Cl" ∧ 
    nh4cl.coefficient = 1 := by
  sorry

end NUMINAMATH_CALUDE_ammonium_chloride_required_l401_40167


namespace NUMINAMATH_CALUDE_video_game_players_l401_40142

theorem video_game_players (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : 
  players_quit = 8 →
  lives_per_player = 6 →
  total_lives = 30 →
  players_quit + (total_lives / lives_per_player) = 13 := by
sorry

end NUMINAMATH_CALUDE_video_game_players_l401_40142


namespace NUMINAMATH_CALUDE_largest_floor_value_l401_40195

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

end NUMINAMATH_CALUDE_largest_floor_value_l401_40195


namespace NUMINAMATH_CALUDE_inequality_proof_l401_40170

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / Real.sqrt (a^2 + 8*b*c) + b / Real.sqrt (b^2 + 8*c*a) + c / Real.sqrt (c^2 + 8*a*b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l401_40170


namespace NUMINAMATH_CALUDE_train_length_calculation_l401_40111

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

end NUMINAMATH_CALUDE_train_length_calculation_l401_40111


namespace NUMINAMATH_CALUDE_inner_square_side_length_l401_40130

/-- A square with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 2) ∧ B = (2, 2) ∧ C = (2, 0) ∧ D = (0, 0))

/-- A smaller square inside the main square -/
structure InnerSquare (outer : Square) :=
  (P Q R S : ℝ × ℝ)
  (P_midpoint : P = (1, 2))
  (S_on_BC : S.1 = 2)
  (is_square : (P.1 - S.1)^2 + (P.2 - S.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2)

/-- The theorem to be proved -/
theorem inner_square_side_length (outer : Square) (inner : InnerSquare outer) :
  Real.sqrt ((inner.P.1 - inner.S.1)^2 + (inner.P.2 - inner.S.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_inner_square_side_length_l401_40130


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l401_40191

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {y | ∃ x, y = -x^2 + 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l401_40191


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l401_40108

/-- Given three real numbers form a geometric progression, prove that the first term is 15 + 5√5 --/
theorem geometric_progression_solution (x : ℝ) : 
  (2*x + 10)^2 = x * (5*x + 10) → x = 15 + 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l401_40108


namespace NUMINAMATH_CALUDE_circle_tangency_theorem_l401_40168

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ -2

-- Define the point T
def point_T : ℝ × ℝ := (4, 0)

-- Define the theorem
theorem circle_tangency_theorem :
  -- Part 1: Prove that curve C is the locus of points equidistant from the centers of M and N
  (∀ x y : ℝ, curve_C x y ↔ 
    (∃ r : ℝ, r > 0 ∧ 
      ((x - (-1))^2 + y^2 = (r + 1)^2) ∧ 
      ((x - 1)^2 + y^2 = (3 - r)^2))) ∧
  -- Part 2: Prove that T(4, 0) satisfies the angle condition
  (∀ k : ℝ, ∀ x₁ y₁ x₂ y₂ : ℝ,
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    y₁ = k * (x₁ - 1) ∧ y₂ = k * (x₂ - 1) →
    (y₁ / (x₁ - 4) + y₂ / (x₂ - 4) = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangency_theorem_l401_40168


namespace NUMINAMATH_CALUDE_square_equality_solutions_l401_40139

theorem square_equality_solutions (x : ℝ) : 
  (x + 1)^2 = (2*x - 1)^2 ↔ x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_square_equality_solutions_l401_40139


namespace NUMINAMATH_CALUDE_fraction_simplification_l401_40192

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) :
  3 / (m^2 - 9) + m / (9 - m^2) = -1 / (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l401_40192


namespace NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l401_40179

theorem average_marks_of_failed_candidates
  (total_candidates : ℕ)
  (overall_average : ℚ)
  (passed_average : ℚ)
  (passed_candidates : ℕ)
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : passed_average = 39)
  (h4 : passed_candidates = 100) :
  let failed_candidates := total_candidates - passed_candidates
  let total_marks := total_candidates * overall_average
  let passed_marks := passed_candidates * passed_average
  let failed_marks := total_marks - passed_marks
  failed_marks / failed_candidates = 15 :=
by sorry

end NUMINAMATH_CALUDE_average_marks_of_failed_candidates_l401_40179


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l401_40112

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > a (n + 1)) ∧
  (∃ r : ℝ, 0 < r ∧ r < 1 ∧ ∀ n, a (n + 1) = r * a n) ∧
  (a 7 * a 14 = 6) ∧
  (a 4 + a 17 = 5)

/-- The main theorem stating the ratio of a_5 to a_18 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 5 / a 18 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l401_40112


namespace NUMINAMATH_CALUDE_unique_right_triangle_existence_l401_40147

/-- A right triangle with leg lengths a and b, and hypotenuse c. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c

/-- The difference between the sum of legs and hypotenuse. -/
def leg_hyp_diff (t : RightTriangle) : ℝ := t.a + t.b - t.c

/-- Theorem: A unique right triangle exists given one leg a and the difference d
    between the sum of the legs and the hypotenuse, if and only if d < a. -/
theorem unique_right_triangle_existence (a d : ℝ) (ha : 0 < a) :
  (∃! t : RightTriangle, t.a = a ∧ leg_hyp_diff t = d) ↔ d < a := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_existence_l401_40147


namespace NUMINAMATH_CALUDE_sean_bought_two_soups_l401_40102

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 1

/-- The number of sodas Sean bought -/
def num_sodas : ℕ := 3

/-- The cost of a single soup in dollars -/
def soup_cost : ℚ := soda_cost * num_sodas

/-- The cost of the sandwich in dollars -/
def sandwich_cost : ℚ := 3 * soup_cost

/-- The total cost of all items in dollars -/
def total_cost : ℚ := 18

/-- The number of soups Sean bought -/
def num_soups : ℕ := 2

theorem sean_bought_two_soups :
  soda_cost * num_sodas + sandwich_cost + soup_cost * num_soups = total_cost :=
sorry

end NUMINAMATH_CALUDE_sean_bought_two_soups_l401_40102


namespace NUMINAMATH_CALUDE_union_complement_equals_less_than_three_l401_40175

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 3}
def B : Set ℝ := {x | x > 2}

-- State the theorem
theorem union_complement_equals_less_than_three :
  A ∪ (univ \ B) = {x : ℝ | x < 3} := by sorry

end NUMINAMATH_CALUDE_union_complement_equals_less_than_three_l401_40175


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l401_40101

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 18 →
    x + 2*x > 18 →
    x + 2*x + 18 ≤ 69 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l401_40101


namespace NUMINAMATH_CALUDE_smallest_integer_x_l401_40189

theorem smallest_integer_x : ∃ x : ℤ, 
  (∀ z : ℤ, (7 - 5*z < 25 ∧ 10 - 3*z > 6) → x ≤ z) ∧ 
  (7 - 5*x < 25 ∧ 10 - 3*x > 6) ∧
  x = -3 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_x_l401_40189


namespace NUMINAMATH_CALUDE_min_value_at_two_l401_40151

/-- The function f(c) = 2c^2 - 8c + 1 attains its minimum value at c = 2 -/
theorem min_value_at_two (c : ℝ) : 
  IsMinOn (fun c => 2 * c^2 - 8 * c + 1) univ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_two_l401_40151


namespace NUMINAMATH_CALUDE_intersection_of_sets_l401_40188

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l401_40188


namespace NUMINAMATH_CALUDE_problem_solution_l401_40121

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : a ^ b = b ^ a) (h4 : b = 27 * a) : a = (27 : ℝ) ^ (1 / 26) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l401_40121


namespace NUMINAMATH_CALUDE_equation_solutions_l401_40153

theorem equation_solutions : 
  {x : ℝ | x^2 - 3 * |x| - 4 = 0} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l401_40153


namespace NUMINAMATH_CALUDE_functional_equation_solution_l401_40118

/-- A complex-valued function satisfying the given functional equation is constant and equal to 1. -/
theorem functional_equation_solution (f : ℂ → ℂ) : 
  (∀ z : ℂ, f z + z * f (1 - z) = 1 + z) → 
  (∀ z : ℂ, f z = 1) := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l401_40118


namespace NUMINAMATH_CALUDE_cube_volume_from_body_diagonal_l401_40103

theorem cube_volume_from_body_diagonal (diagonal : ℝ) (h : diagonal = 15) :
  ∃ (side : ℝ), side * Real.sqrt 3 = diagonal ∧ side^3 = 375 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_body_diagonal_l401_40103


namespace NUMINAMATH_CALUDE_power_sum_difference_l401_40166

theorem power_sum_difference : 2^(1+2+3) - (2^1 + 2^2 + 2^3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_difference_l401_40166


namespace NUMINAMATH_CALUDE_function_difference_inequality_l401_40114

theorem function_difference_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h1 : ∀ x > 1, deriv f x > deriv g x)
  (h2 : ∀ x < 1, deriv f x < deriv g x) :
  f 2 - f 1 > g 2 - g 1 :=
by sorry

end NUMINAMATH_CALUDE_function_difference_inequality_l401_40114


namespace NUMINAMATH_CALUDE_smallest_gcd_l401_40169

theorem smallest_gcd (p q r : ℕ+) (h1 : Nat.gcd p q = 294) (h2 : Nat.gcd p r = 847) :
  ∃ (q' r' : ℕ+), Nat.gcd q' r' = 49 ∧ 
    ∀ (q'' r'' : ℕ+), Nat.gcd p q'' = 294 → Nat.gcd p r'' = 847 → 
      Nat.gcd q'' r'' ≥ 49 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_l401_40169


namespace NUMINAMATH_CALUDE_f_max_min_difference_l401_40197

noncomputable def f (x : ℝ) : ℝ := Real.exp (Real.sin x + Real.cos x) - (1/2) * Real.sin (2 * x)

theorem f_max_min_difference :
  (⨆ (x : ℝ), f x) - (⨅ (x : ℝ), f x) = Real.exp (Real.sqrt 2) - Real.exp (-Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_difference_l401_40197


namespace NUMINAMATH_CALUDE_train_average_speed_l401_40193

/-- 
Given a train that travels two distances in two time periods, 
this theorem proves that its average speed is the total distance divided by the total time.
-/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) 
  (h2 : time1 = 3.5)
  (h3 : distance2 = 470)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 106 := by
sorry

end NUMINAMATH_CALUDE_train_average_speed_l401_40193


namespace NUMINAMATH_CALUDE_real_equal_roots_condition_l401_40113

theorem real_equal_roots_condition (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x + 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y + 2 * y + 12 = 0 → y = x) ↔ 
  (k = -10 ∨ k = 14) := by sorry

end NUMINAMATH_CALUDE_real_equal_roots_condition_l401_40113


namespace NUMINAMATH_CALUDE_unique_solution_condition_l401_40161

/-- The equation has exactly one solution if and only if a is in the specified set -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, a * |2 + x| + (x^2 + x - 12) / (x + 4) = 0) ↔ 
  (a ∈ Set.Ioc (-1) 1 ∪ {7/2}) := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l401_40161


namespace NUMINAMATH_CALUDE_square_side_length_with_inscribed_circle_l401_40124

theorem square_side_length_with_inscribed_circle (s : ℝ) : 
  (4 * s = π * (s / 2)^2) → s = 16 / π := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_with_inscribed_circle_l401_40124


namespace NUMINAMATH_CALUDE_max_blocks_is_twelve_l401_40138

/-- A block covers exactly two cells -/
structure Block where
  cells : Fin 16 → Fin 16
  covers_two : ∃ (c1 c2 : Fin 16), c1 ≠ c2 ∧ (∀ c, cells c = c1 ∨ cells c = c2)

/-- Configuration of blocks on a 4x4 grid -/
structure Configuration where
  blocks : List Block
  all_cells_covered : ∀ c : Fin 16, ∃ b ∈ blocks, ∃ c', b.cells c' = c
  removal_uncovers : ∀ b ∈ blocks, ∃ c : Fin 16, (∀ b' ∈ blocks, b' ≠ b → ∀ c', b'.cells c' ≠ c)

/-- The maximum number of blocks in a valid configuration -/
def max_blocks : ℕ := 12

/-- The theorem stating that 12 is the maximum number of blocks -/
theorem max_blocks_is_twelve :
  ∀ cfg : Configuration, cfg.blocks.length ≤ max_blocks :=
sorry

end NUMINAMATH_CALUDE_max_blocks_is_twelve_l401_40138


namespace NUMINAMATH_CALUDE_hide_and_seek_players_l401_40160

-- Define variables for each person
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) :=
sorry

end NUMINAMATH_CALUDE_hide_and_seek_players_l401_40160


namespace NUMINAMATH_CALUDE_absolute_value_ratio_l401_40143

theorem absolute_value_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a^2 + b^2 = 8*a*b) :
  |a + b| / |a - b| = Real.sqrt 15 / 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_ratio_l401_40143


namespace NUMINAMATH_CALUDE_sequence_inequality_l401_40174

theorem sequence_inequality (a : ℕ → ℝ) (h_nonneg : ∀ n, a n ≥ 0)
  (h_ineq : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m > 0 → n ≥ m → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l401_40174


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l401_40140

theorem sunglasses_cap_probability 
  (total_sunglasses : ℕ) 
  (total_caps : ℕ) 
  (total_hats : ℕ) 
  (prob_cap_and_sunglasses : ℚ) 
  (h1 : total_sunglasses = 80) 
  (h2 : total_caps = 60) 
  (h3 : total_hats = 40) 
  (h4 : prob_cap_and_sunglasses = 1/3) :
  (total_caps * prob_cap_and_sunglasses) / total_sunglasses = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l401_40140


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l401_40141

def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/2) (h₂ : d = 1/2) :
  arithmeticSequence a₁ d 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l401_40141


namespace NUMINAMATH_CALUDE_black_marble_probability_l401_40171

/-- The probability of drawing a black marble from a bag -/
theorem black_marble_probability 
  (yellow : ℕ) 
  (blue : ℕ) 
  (green : ℕ) 
  (black : ℕ) 
  (h_yellow : yellow = 12) 
  (h_blue : blue = 10) 
  (h_green : green = 5) 
  (h_black : black = 1) : 
  (black : ℚ) / (yellow + blue + green + black : ℚ) = 1 / 28 := by
  sorry

end NUMINAMATH_CALUDE_black_marble_probability_l401_40171


namespace NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l401_40196

/-- The number of ways to insert 3 distinct objects into 11 spaces,
    such that no two inserted objects are adjacent. -/
def insert_non_adjacent (n m : ℕ) : ℕ :=
  Nat.descFactorial (n + 1) m

theorem spring_festival_gala_arrangements : insert_non_adjacent 10 3 = 990 := by
  sorry

end NUMINAMATH_CALUDE_spring_festival_gala_arrangements_l401_40196
