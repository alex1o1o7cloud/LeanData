import Mathlib

namespace NUMINAMATH_CALUDE_abc_books_sold_l1807_180761

theorem abc_books_sold (top_price : ℕ) (abc_price : ℕ) (top_sold : ℕ) (earnings_diff : ℕ) :
  top_price = 8 →
  abc_price = 23 →
  top_sold = 13 →
  earnings_diff = 12 →
  ∃ (abc_sold : ℕ), abc_sold * abc_price = top_sold * top_price - earnings_diff :=
by
  sorry

end NUMINAMATH_CALUDE_abc_books_sold_l1807_180761


namespace NUMINAMATH_CALUDE_remainder_3_pow_2003_mod_13_l1807_180716

theorem remainder_3_pow_2003_mod_13 :
  ∃ k : ℤ, 3^2003 = 13 * k + 9 :=
by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2003_mod_13_l1807_180716


namespace NUMINAMATH_CALUDE_additional_three_pointers_l1807_180752

def points_to_tie : ℕ := 17
def points_over_record : ℕ := 5
def old_record : ℕ := 257
def free_throws : ℕ := 5
def regular_baskets : ℕ := 4
def normal_three_pointers : ℕ := 2

def points_per_free_throw : ℕ := 1
def points_per_regular_basket : ℕ := 2
def points_per_three_pointer : ℕ := 3

def total_points_final_game : ℕ := points_to_tie + points_over_record
def points_from_free_throws : ℕ := free_throws * points_per_free_throw
def points_from_regular_baskets : ℕ := regular_baskets * points_per_regular_basket
def points_from_three_pointers : ℕ := total_points_final_game - points_from_free_throws - points_from_regular_baskets

theorem additional_three_pointers (
  h1 : points_from_three_pointers % points_per_three_pointer = 0
) : (points_from_three_pointers / points_per_three_pointer) - normal_three_pointers = 1 := by
  sorry

end NUMINAMATH_CALUDE_additional_three_pointers_l1807_180752


namespace NUMINAMATH_CALUDE_eleven_integer_chords_l1807_180775

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distance_to_p : ℝ

/-- Counts the number of integer-length chords containing P -/
def count_integer_chords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem eleven_integer_chords :
  let c := CircleWithPoint.mk 17 12
  count_integer_chords c = 11 := by sorry

end NUMINAMATH_CALUDE_eleven_integer_chords_l1807_180775


namespace NUMINAMATH_CALUDE_regression_line_intercept_l1807_180791

/-- Prove that a regression line with slope 1.23 passing through (4, 5) has y-intercept 0.08 -/
theorem regression_line_intercept (slope : ℝ) (center_x center_y : ℝ) (y_intercept : ℝ) : 
  slope = 1.23 → center_x = 4 → center_y = 5 → 
  center_y = slope * center_x + y_intercept →
  y_intercept = 0.08 := by
sorry

end NUMINAMATH_CALUDE_regression_line_intercept_l1807_180791


namespace NUMINAMATH_CALUDE_right_triangle_tan_b_l1807_180700

theorem right_triangle_tan_b (A B C : ℝ) (h1 : C = π / 2) (h2 : Real.cos A = 3 / 5) : 
  Real.tan B = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_tan_b_l1807_180700


namespace NUMINAMATH_CALUDE_min_value_x_one_minus_y_l1807_180711

theorem min_value_x_one_minus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 4 * x^2 + 4 * x * y + y^2 + 2 * x + y - 6 = 0) :
  ∀ z : ℝ, z > 0 → ∀ w : ℝ, w > 0 →
  4 * z^2 + 4 * z * w + w^2 + 2 * z + w - 6 = 0 →
  x * (1 - y) ≤ z * (1 - w) ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
  4 * a^2 + 4 * a * b + b^2 + 2 * a + b - 6 = 0 ∧
  a * (1 - b) = -1/8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_one_minus_y_l1807_180711


namespace NUMINAMATH_CALUDE_expression_simplification_l1807_180773

theorem expression_simplification (x : ℝ) : 
  x - 3*(1+x) + 4*(1-x)^2 - 5*(1+3*x) = 4*x^2 - 25*x - 4 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l1807_180773


namespace NUMINAMATH_CALUDE_train_length_l1807_180787

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 144 →
  time_s = 8.7493 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 350 := by
sorry

end NUMINAMATH_CALUDE_train_length_l1807_180787


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1807_180741

def I : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_of_M_and_N : M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1807_180741


namespace NUMINAMATH_CALUDE_theresa_final_week_hours_l1807_180704

/-- The number of weeks Theresa needs to work -/
def total_weeks : ℕ := 6

/-- The required average hours per week -/
def required_average : ℕ := 12

/-- The hours worked in the first five weeks -/
def first_five_weeks : List ℕ := [10, 13, 9, 14, 11]

/-- The sum of hours worked in the first five weeks -/
def sum_first_five : ℕ := first_five_weeks.sum

/-- The number of hours Theresa needs to work in the final week -/
def final_week_hours : ℕ := 15

theorem theresa_final_week_hours :
  (sum_first_five + final_week_hours) / total_weeks = required_average :=
sorry

end NUMINAMATH_CALUDE_theresa_final_week_hours_l1807_180704


namespace NUMINAMATH_CALUDE_calendar_sum_property_l1807_180744

/-- Represents a monthly calendar with dates behind letters --/
structure Calendar where
  x : ℕ  -- The date behind C
  dateA : ℕ := x + 1
  dateB : ℕ := x + 13
  dateP : ℕ := x + 14
  dateQ : ℕ
  dateR : ℕ
  dateS : ℕ
  dateT : ℕ

/-- The letter P is the only one that satisfies the condition --/
theorem calendar_sum_property (cal : Calendar) :
  (cal.x + cal.dateP = cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateQ ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateR ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateS ≠ cal.dateA + cal.dateB) ∧
  (cal.x + cal.dateT ≠ cal.dateA + cal.dateB) :=
by sorry

end NUMINAMATH_CALUDE_calendar_sum_property_l1807_180744


namespace NUMINAMATH_CALUDE_red_other_side_probability_l1807_180729

structure Card where
  side1 : Bool  -- True for red, False for black
  side2 : Bool

def box : Finset Card := sorry

axiom box_size : box.card = 8

axiom black_both_sides : (box.filter (fun c => !c.side1 ∧ !c.side2)).card = 4

axiom black_red_sides : (box.filter (fun c => (c.side1 ∧ !c.side2) ∨ (!c.side1 ∧ c.side2))).card = 2

axiom red_both_sides : (box.filter (fun c => c.side1 ∧ c.side2)).card = 2

def observe_red (c : Card) : Bool := c.side1 ∨ c.side2

def other_side_red (c : Card) : Bool := c.side1 ∧ c.side2

theorem red_other_side_probability :
  (box.filter (fun c => other_side_red c)).card / (box.filter (fun c => observe_red c)).card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_red_other_side_probability_l1807_180729


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l1807_180797

/-- A random variable following a normal distribution with mean 2 and some variance σ² -/
noncomputable def ξ : Real → ℝ := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : ℝ → ℝ := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : ℝ → ℝ := sorry

/-- The condition that P(ξ < 4) = 0.8 -/
axiom cdf_at_4 : cdf_ξ 4 = 0.8

/-- The theorem to prove -/
theorem normal_distribution_probability :
  cdf_ξ 2 - cdf_ξ 0 = 0.3 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l1807_180797


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1807_180779

theorem consecutive_integers_sum_of_cubes (n : ℤ) : 
  n > 0 ∧ (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 11534 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 74836 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_cubes_l1807_180779


namespace NUMINAMATH_CALUDE_haley_trees_died_l1807_180709

/-- The number of trees that died due to a typhoon -/
def trees_died (initial_trees : ℕ) (remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 2 trees died in Haley's backyard after the typhoon -/
theorem haley_trees_died : trees_died 12 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_died_l1807_180709


namespace NUMINAMATH_CALUDE_problem_solution_l1807_180748

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -3 ∨ (23 ≤ x ∧ x < 27))
  (h2 : a < b) :
  a + 2*b + 3*c = 71 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1807_180748


namespace NUMINAMATH_CALUDE_jason_career_percentage_increase_l1807_180792

/-- Represents the career progression of a military person --/
structure MilitaryCareer where
  join_age : ℕ
  years_to_chief : ℕ
  retirement_age : ℕ
  years_after_master_chief : ℕ

/-- Calculates the percentage increase in time from chief to master chief
    compared to the time to become a chief --/
def percentage_increase (career : MilitaryCareer) : ℚ :=
  let total_years := career.retirement_age - career.join_age
  let years_chief_to_retirement := total_years - career.years_to_chief
  let years_to_master_chief := years_chief_to_retirement - career.years_after_master_chief
  (years_to_master_chief - career.years_to_chief) / career.years_to_chief * 100

/-- Theorem stating that for Jason's career, the percentage increase is 25% --/
theorem jason_career_percentage_increase :
  let jason_career := MilitaryCareer.mk 18 8 46 10
  percentage_increase jason_career = 25 := by
  sorry

end NUMINAMATH_CALUDE_jason_career_percentage_increase_l1807_180792


namespace NUMINAMATH_CALUDE_alloy_mixture_l1807_180747

/-- The amount of alloy B mixed with alloy A -/
def amount_alloy_B : ℝ := 180

/-- The amount of alloy A -/
def amount_alloy_A : ℝ := 120

/-- The ratio of lead to tin in alloy A -/
def ratio_A : ℚ := 2 / 3

/-- The ratio of tin to copper in alloy B -/
def ratio_B : ℚ := 3 / 5

/-- The amount of tin in the new alloy -/
def amount_tin_new : ℝ := 139.5

theorem alloy_mixture :
  amount_alloy_B = 180 ∧
  (ratio_A * amount_alloy_A + ratio_B * amount_alloy_B) / (1 + ratio_A) = amount_tin_new :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_l1807_180747


namespace NUMINAMATH_CALUDE_race_probability_l1807_180763

theorem race_probability (total_cars : ℕ) (prob_X prob_Z prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Z = 1/8 →
  prob_total = 0.39166666666666666 →
  ∃ (prob_Y : ℝ), prob_Y = prob_total - prob_X - prob_Z ∧ prob_Y = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l1807_180763


namespace NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1807_180715

-- Define the line Ax + By + C = 0
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (l : Line) : Prop :=
  l.A * l.C < 0 ∧ l.B * l.C > 0

-- Define the second quadrant
def in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

-- Theorem statement
theorem line_not_in_second_quadrant (l : Line) 
  (h : satisfies_conditions l) :
  ¬∃ (x y : ℝ), l.A * x + l.B * y + l.C = 0 ∧ in_second_quadrant x y :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_second_quadrant_l1807_180715


namespace NUMINAMATH_CALUDE_sally_monday_shirts_l1807_180717

def shirts_sewn_tuesday : ℕ := 3
def shirts_sewn_wednesday : ℕ := 2
def buttons_per_shirt : ℕ := 5
def total_buttons_needed : ℕ := 45

theorem sally_monday_shirts :
  ∃ (monday_shirts : ℕ),
    monday_shirts + shirts_sewn_tuesday + shirts_sewn_wednesday = 
    total_buttons_needed / buttons_per_shirt ∧
    monday_shirts = 4 := by
  sorry

end NUMINAMATH_CALUDE_sally_monday_shirts_l1807_180717


namespace NUMINAMATH_CALUDE_system_of_equations_sum_l1807_180758

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 13 * x + b * y + c * z = 0)
  (eq2 : a * x + 23 * y + c * z = 0)
  (eq3 : a * x + b * y + 42 * z = 0)
  (ha : a ≠ 13)
  (hx : x ≠ 0) :
  13 / (a - 13) + 23 / (b - 23) + 42 / (c - 42) = -2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_sum_l1807_180758


namespace NUMINAMATH_CALUDE_only_molality_can_be_calculated_l1807_180769

-- Define the given quantities
variable (mass_solute : ℝ)
variable (mass_solvent : ℝ)
variable (molar_mass_solute : ℝ)
variable (molar_mass_solvent : ℝ)

-- Define the quantitative descriptions
def can_calculate_molarity (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

def can_calculate_molality (mass_solute molar_mass_solute mass_solvent : ℝ) : Prop :=
  mass_solvent > 0 ∧ molar_mass_solute > 0

def can_calculate_density (mass_solute mass_solvent : ℝ) : Prop :=
  ∃ (volume_solution : ℝ), volume_solution > 0

-- Theorem statement
theorem only_molality_can_be_calculated
  (mass_solute mass_solvent molar_mass_solute molar_mass_solvent : ℝ) :
  can_calculate_molality mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_molarity mass_solute molar_mass_solute mass_solvent ∧
  ¬can_calculate_density mass_solute mass_solvent :=
sorry

end NUMINAMATH_CALUDE_only_molality_can_be_calculated_l1807_180769


namespace NUMINAMATH_CALUDE_funfair_tickets_l1807_180736

theorem funfair_tickets (total_rolls : ℕ) (fourth_grade_percent : ℚ) 
  (fifth_grade_percent : ℚ) (sixth_grade_bought : ℕ) (tickets_left : ℕ) :
  total_rolls = 30 →
  fourth_grade_percent = 30 / 100 →
  fifth_grade_percent = 50 / 100 →
  sixth_grade_bought = 100 →
  tickets_left = 950 →
  ∃ (tickets_per_roll : ℕ),
    tickets_per_roll * total_rolls * (1 - fourth_grade_percent) * (1 - fifth_grade_percent) - sixth_grade_bought = tickets_left ∧
    tickets_per_roll = 100 := by
  sorry

end NUMINAMATH_CALUDE_funfair_tickets_l1807_180736


namespace NUMINAMATH_CALUDE_ratio_problem_l1807_180732

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.2 * b) : m / x = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1807_180732


namespace NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1807_180703

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line l₁: x + ay + 3 = 0 -/
def l1 (a : ℝ) : Line :=
  { a := 1, b := a, c := 3 }

/-- The second line l₂: (a-2)x + 3y + a = 0 -/
def l2 (a : ℝ) : Line :=
  { a := a - 2, b := 3, c := a }

/-- Theorem: The lines l₁ and l₂ are parallel if and only if a = -1 -/
theorem lines_parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, are_parallel (l1 a) (l2 a) ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_lines_parallel_iff_a_eq_neg_one_l1807_180703


namespace NUMINAMATH_CALUDE_exponential_function_not_multiplicative_l1807_180798

theorem exponential_function_not_multiplicative : ¬∀ a b : ℝ, Real.exp (a * b) = Real.exp a * Real.exp b := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_not_multiplicative_l1807_180798


namespace NUMINAMATH_CALUDE_gcd_390_455_l1807_180750

theorem gcd_390_455 : Nat.gcd 390 455 = 65 := by sorry

end NUMINAMATH_CALUDE_gcd_390_455_l1807_180750


namespace NUMINAMATH_CALUDE_find_number_l1807_180782

theorem find_number (x : ℝ) : ((4 * x - 28) / 7 + 12 = 36) → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1807_180782


namespace NUMINAMATH_CALUDE_book_price_increase_l1807_180777

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  new_price = 390 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = new_price →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l1807_180777


namespace NUMINAMATH_CALUDE_prob_fifth_six_given_two_sixes_l1807_180719

/-- Represents a six-sided die -/
inductive Die
| Fair
| Biased

/-- Probability of rolling a six for a given die -/
def prob_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 1/2

/-- Probability of rolling a number other than six for a given die -/
def prob_not_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/10

/-- Probability of rolling at least two sixes in four rolls for a given die -/
def prob_at_least_two_sixes (d : Die) : ℚ :=
  match d with
  | Die.Fair => 11/1296
  | Die.Biased => 11/16

/-- The main theorem -/
theorem prob_fifth_six_given_two_sixes (d : Die) : 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) *
  (prob_six d * prob_at_least_two_sixes d) / 
  (prob_at_least_two_sixes Die.Fair + prob_at_least_two_sixes Die.Biased) = 325/656 :=
sorry

end NUMINAMATH_CALUDE_prob_fifth_six_given_two_sixes_l1807_180719


namespace NUMINAMATH_CALUDE_jills_peaches_l1807_180786

/-- Given that Steven has 19 peaches and 13 more peaches than Jill,
    prove that Jill has 6 peaches. -/
theorem jills_peaches (steven_peaches : ℕ) (steven_jill_diff : ℕ) 
  (h1 : steven_peaches = 19)
  (h2 : steven_peaches = steven_jill_diff + jill_peaches) :
  jill_peaches = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_jills_peaches_l1807_180786


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1807_180795

/-- Given two parallel vectors a = (-3, 2) and b = (1, x), prove that x = -2/3 -/
theorem parallel_vectors_x_value (x : ℚ) : 
  let a : ℚ × ℚ := (-3, 2)
  let b : ℚ × ℚ := (1, x)
  (∃ (k : ℚ), k ≠ 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2) → x = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1807_180795


namespace NUMINAMATH_CALUDE_banana_slices_per_yogurt_l1807_180723

/-- Given that one banana yields 10 slices, 5 yogurts need to be made, and 4 bananas are bought,
    prove that 8 banana slices are needed for each yogurt. -/
theorem banana_slices_per_yogurt :
  let slices_per_banana : ℕ := 10
  let yogurts_to_make : ℕ := 5
  let bananas_bought : ℕ := 4
  let total_slices : ℕ := slices_per_banana * bananas_bought
  let slices_per_yogurt : ℕ := total_slices / yogurts_to_make
  slices_per_yogurt = 8 := by sorry

end NUMINAMATH_CALUDE_banana_slices_per_yogurt_l1807_180723


namespace NUMINAMATH_CALUDE_find_number_l1807_180702

theorem find_number : ∃ x : ℝ, 3 * x + 3 * 14 + 3 * 18 + 11 = 152 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1807_180702


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1807_180725

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a where a_4 * a_6 = 5, prove that a_2 * a_3 * a_7 * a_8 = 25 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : IsGeometricSequence a) 
    (h_prod : a 4 * a 6 = 5) : a 2 * a 3 * a 7 * a 8 = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1807_180725


namespace NUMINAMATH_CALUDE_function_satisfies_conditions_l1807_180780

-- Define the function
def f (x : ℝ) : ℝ := -2 * x^2 + 3 * x

-- State the theorem
theorem function_satisfies_conditions :
  (f 1 = 1) ∧ 
  (∃ x y, x > 0 ∧ y < 0 ∧ f x = y) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_conditions_l1807_180780


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1807_180766

theorem min_value_on_circle (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (min : ℝ), min = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1807_180766


namespace NUMINAMATH_CALUDE_wardrobe_probability_l1807_180730

def num_shirts : ℕ := 5
def num_shorts : ℕ := 6
def num_socks : ℕ := 7
def num_selected : ℕ := 4

def total_articles : ℕ := num_shirts + num_shorts + num_socks

theorem wardrobe_probability :
  (Nat.choose num_shirts 2 * Nat.choose num_shorts 1 * Nat.choose num_socks 1) /
  (Nat.choose total_articles num_selected) = 7 / 51 :=
by sorry

end NUMINAMATH_CALUDE_wardrobe_probability_l1807_180730


namespace NUMINAMATH_CALUDE_relay_team_selection_l1807_180727

/-- The number of athletes in the track and field team -/
def total_athletes : ℕ := 16

/-- The number of triplets -/
def num_triplets : ℕ := 3

/-- The number of twins -/
def num_twins : ℕ := 2

/-- The size of the relay team -/
def team_size : ℕ := 7

/-- The number of ways to choose the relay team -/
def num_ways : ℕ := 3762

theorem relay_team_selection :
  (num_triplets * (num_twins * (Nat.choose (total_athletes - num_triplets - 1 - 1) (team_size - 1 - 1)) +
  1 * (Nat.choose (total_athletes - num_triplets - 2) (team_size - 1 - 2)))) = num_ways :=
sorry

end NUMINAMATH_CALUDE_relay_team_selection_l1807_180727


namespace NUMINAMATH_CALUDE_saturday_duty_probability_is_one_sixth_l1807_180708

/-- A person's weekly night duty schedule -/
structure DutySchedule where
  total_duties : ℕ
  sunday_duty : Bool
  h_total : total_duties = 2
  h_sunday : sunday_duty = true

/-- The probability of being on duty on Saturday night given the duty schedule -/
def saturday_duty_probability (schedule : DutySchedule) : ℚ :=
  1 / 6

/-- Theorem stating that the probability of Saturday duty is 1/6 -/
theorem saturday_duty_probability_is_one_sixth (schedule : DutySchedule) :
  saturday_duty_probability schedule = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_saturday_duty_probability_is_one_sixth_l1807_180708


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1807_180785

theorem binomial_coefficient_divisibility (p : ℕ) (hp : Nat.Prime p) :
  (Nat.choose (2 * p) p - 2) % (p^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l1807_180785


namespace NUMINAMATH_CALUDE_james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l1807_180751

/-- The sum of James and Louise's current ages given the conditions -/
theorem james_and_louise_age_sum : ℝ :=
  let james_age : ℝ := sorry
  let louise_age : ℝ := sorry

  -- James is eight years older than Louise
  have h1 : james_age = louise_age + 8 := by sorry

  -- Ten years from now, James will be five times as old as Louise was five years ago
  have h2 : james_age + 10 = 5 * (louise_age - 5) := by sorry

  -- The sum of their current ages
  have h3 : james_age + louise_age = 29.5 := by sorry

  29.5

theorem james_and_louise_age_sum_is_correct : james_and_louise_age_sum = 29.5 := by sorry

end NUMINAMATH_CALUDE_james_and_louise_age_sum_james_and_louise_age_sum_is_correct_l1807_180751


namespace NUMINAMATH_CALUDE_games_for_23_teams_l1807_180731

/-- A single-elimination tournament where teams are eliminated after one loss and no ties are possible. -/
structure Tournament :=
  (num_teams : ℕ)

/-- The number of games needed to declare a champion in a single-elimination tournament. -/
def games_to_champion (t : Tournament) : ℕ := t.num_teams - 1

/-- Theorem: In a single-elimination tournament with 23 teams, 22 games are needed to declare a champion. -/
theorem games_for_23_teams :
  ∀ t : Tournament, t.num_teams = 23 → games_to_champion t = 22 := by
  sorry


end NUMINAMATH_CALUDE_games_for_23_teams_l1807_180731


namespace NUMINAMATH_CALUDE_problem_solution_l1807_180749

theorem problem_solution (a b c d : ℝ) : 
  a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c + 4 - d) → d = 17/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1807_180749


namespace NUMINAMATH_CALUDE_function_value_at_inverse_l1807_180781

/-- Given a function f(x) = kx + 2/x^3 - 3 where k is a real number,
    if f(ln 6) = 1, then f(ln(1/6)) = -7 -/
theorem function_value_at_inverse (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ k * x + 2 / x^3 - 3
  f (Real.log 6) = 1 → f (Real.log (1/6)) = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_inverse_l1807_180781


namespace NUMINAMATH_CALUDE_open_box_volume_is_5120_l1807_180712

/-- The volume of an open box formed by cutting squares from a rectangular sheet. -/
def open_box_volume (sheet_length sheet_width cut_side : ℝ) : ℝ :=
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side

/-- Theorem: The volume of the open box is 5120 m³ -/
theorem open_box_volume_is_5120 :
  open_box_volume 48 36 8 = 5120 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_is_5120_l1807_180712


namespace NUMINAMATH_CALUDE_total_length_climbed_50_30_6_25_l1807_180784

/-- The total length of ladders climbed by two workers in centimeters -/
def total_length_climbed (keaton_ladder_height : ℕ) (keaton_climbs : ℕ) 
  (reece_ladder_diff : ℕ) (reece_climbs : ℕ) : ℕ :=
  let reece_ladder_height := keaton_ladder_height - reece_ladder_diff
  let keaton_total := keaton_ladder_height * keaton_climbs
  let reece_total := reece_ladder_height * reece_climbs
  (keaton_total + reece_total) * 100

/-- Theorem stating the total length climbed by both workers -/
theorem total_length_climbed_50_30_6_25 :
  total_length_climbed 50 30 6 25 = 260000 := by
  sorry

end NUMINAMATH_CALUDE_total_length_climbed_50_30_6_25_l1807_180784


namespace NUMINAMATH_CALUDE_solve_cube_equation_l1807_180788

theorem solve_cube_equation : ∃ x : ℝ, (x - 5)^3 = -(1/27)⁻¹ ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_cube_equation_l1807_180788


namespace NUMINAMATH_CALUDE_set_intersection_equality_l1807_180793

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | |x - 1| > 2}

-- Define set B
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- Define the open interval (2, 3]
def open_interval : Set ℝ := Set.Ioc 2 3

-- Theorem statement
theorem set_intersection_equality :
  (Set.compl A ∩ B) = open_interval :=
sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l1807_180793


namespace NUMINAMATH_CALUDE_career_preference_degrees_l1807_180757

/-- Represents the ratio of male to female students in a class -/
structure GenderRatio where
  male : ℕ
  female : ℕ

/-- Represents the number of students preferring a career -/
structure CareerPreference where
  male : ℕ
  female : ℕ

/-- Calculates the degrees in a circle graph for a career preference -/
def degreesForPreference (ratio : GenderRatio) (pref : CareerPreference) : ℚ :=
  360 * (pref.male + pref.female : ℚ) / (ratio.male + ratio.female : ℚ)

theorem career_preference_degrees 
  (ratio : GenderRatio) 
  (pref : CareerPreference) : 
  ratio.male = 2 ∧ ratio.female = 3 ∧ pref.male = 1 ∧ pref.female = 1 → 
  degreesForPreference ratio pref = 144 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_degrees_l1807_180757


namespace NUMINAMATH_CALUDE_remainder_theorem_l1807_180720

theorem remainder_theorem (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1807_180720


namespace NUMINAMATH_CALUDE_tangent_lines_proof_l1807_180759

-- Define the curves
def f (x : ℝ) : ℝ := x^3 + x^2 + 1
def g (x : ℝ) : ℝ := x^2

-- Define the points
def P1 : ℝ × ℝ := (-1, 1)
def P2 : ℝ × ℝ := (3, 5)

-- Define the tangent line equations
def tangent_line1 (x y : ℝ) : Prop := x - y + 2 = 0
def tangent_line2 (x y : ℝ) : Prop := 2*x - y - 1 = 0
def tangent_line3 (x y : ℝ) : Prop := 10*x - y - 25 = 0

theorem tangent_lines_proof :
  (∀ x y : ℝ, y = f x → (x, y) = P1 → tangent_line1 x y) ∧
  (∀ x y : ℝ, y = g x → (x, y) = P2 → (tangent_line2 x y ∨ tangent_line3 x y)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_proof_l1807_180759


namespace NUMINAMATH_CALUDE_convex_polyhedron_same_sided_faces_l1807_180735

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  faces : Set Face

/-- Theorem: Every convex polyhedron has at least two faces with the same number of sides -/
theorem convex_polyhedron_same_sided_faces (P : ConvexPolyhedron) :
  ∃ (f₁ f₂ : Face), f₁ ∈ P.faces ∧ f₂ ∈ P.faces ∧ f₁ ≠ f₂ ∧ f₁.sides = f₂.sides :=
sorry

end NUMINAMATH_CALUDE_convex_polyhedron_same_sided_faces_l1807_180735


namespace NUMINAMATH_CALUDE_parabola_translation_l1807_180705

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk (1/2) 0 1
  let translated := translate original 1 (-3)
  y = 1/2 * x^2 + 1 → y = 1/2 * (x-1)^2 - 2 := by
  sorry


end NUMINAMATH_CALUDE_parabola_translation_l1807_180705


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1807_180713

theorem quadratic_inequality (x : ℝ) : -9*x^2 + 6*x + 15 > 0 ↔ -1 < x ∧ x < 5/3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1807_180713


namespace NUMINAMATH_CALUDE_opposite_colors_in_prism_l1807_180794

-- Define the set of colors
inductive Color
  | Red
  | Yellow
  | Blue
  | Black
  | White
  | Green

-- Define a cube as a function from faces to colors
def Cube := Fin 6 → Color

-- Define the property of having all different colors
def allDifferentColors (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c i ≠ c j

-- Define the property of opposite faces having the same color in a rectangular prism
def oppositeColorsSame (c : Cube) : Prop :=
  (c 0 = Color.Red ∧ c 5 = Color.Green) ∨
  (c 0 = Color.Green ∧ c 5 = Color.Red) ∧
  (c 1 = Color.Yellow ∧ c 4 = Color.Blue) ∨
  (c 1 = Color.Blue ∧ c 4 = Color.Yellow) ∧
  (c 2 = Color.Black ∧ c 3 = Color.White) ∨
  (c 2 = Color.White ∧ c 3 = Color.Black)

-- Theorem stating the opposite colors in the rectangular prism
theorem opposite_colors_in_prism (c : Cube) 
  (h1 : allDifferentColors c) 
  (h2 : oppositeColorsSame c) :
  (c 0 = Color.Red → c 5 = Color.Green) ∧
  (c 1 = Color.Yellow → c 4 = Color.Blue) ∧
  (c 2 = Color.Black → c 3 = Color.White) :=
by sorry

end NUMINAMATH_CALUDE_opposite_colors_in_prism_l1807_180794


namespace NUMINAMATH_CALUDE_A_and_D_mutually_exclusive_but_not_complementary_l1807_180726

-- Define the sample space for a fair six-sided die
def DieOutcome := Fin 6

-- Define the events
def event_A (n : DieOutcome) : Prop := n.val % 2 = 1
def event_B (n : DieOutcome) : Prop := n.val % 2 = 0
def event_C (n : DieOutcome) : Prop := n.val % 2 = 0
def event_D (n : DieOutcome) : Prop := n.val = 2 ∨ n.val = 4

-- Define mutual exclusivity
def mutually_exclusive (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, ¬(e1 n ∧ e2 n)

-- Define complementary events
def complementary (e1 e2 : DieOutcome → Prop) : Prop :=
  ∀ n : DieOutcome, e1 n ↔ ¬e2 n

-- Theorem to prove
theorem A_and_D_mutually_exclusive_but_not_complementary :
  mutually_exclusive event_A event_D ∧ ¬complementary event_A event_D :=
sorry

end NUMINAMATH_CALUDE_A_and_D_mutually_exclusive_but_not_complementary_l1807_180726


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1807_180760

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x^2 * Real.cos (4 / (3 * x)) + x^2 / 2 else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1807_180760


namespace NUMINAMATH_CALUDE_range_of_a_l1807_180740

-- Define the conditions
def condition_p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def condition_q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, condition_q x → condition_p x a) →
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1807_180740


namespace NUMINAMATH_CALUDE_correct_mark_proof_l1807_180755

/-- Proves that given a class of 20 pupils, if entering 73 instead of the correct mark
    increases the class average by 0.5, then the correct mark should have been 63. -/
theorem correct_mark_proof (n : ℕ) (wrong_mark correct_mark : ℝ) 
    (h1 : n = 20)
    (h2 : wrong_mark = 73)
    (h3 : (wrong_mark - correct_mark) / n = 0.5) :
  correct_mark = 63 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_proof_l1807_180755


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l1807_180722

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : Finset.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l1807_180722


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1807_180743

theorem complex_fraction_equality : (3 - Complex.I) / (1 - Complex.I) = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1807_180743


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1807_180771

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 
    4 * sin x * cos (π/2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3*π/2 - x) * cos (π + x) = 1 ↔ 
    (∃ k : ℤ, x = arctan (1/3) + π * k) ∨ (∃ n : ℤ, x = π/4 + π * n) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1807_180771


namespace NUMINAMATH_CALUDE_cube_edge_length_range_l1807_180799

theorem cube_edge_length_range (V : ℝ) (a : ℝ) (h1 : V = 9) (h2 : V = a^3) :
  2 < a ∧ a < 2.5 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_range_l1807_180799


namespace NUMINAMATH_CALUDE_diagonal_length_is_2_8_l1807_180753

/-- Represents a quadrilateral with given side lengths and a diagonal -/
structure Quadrilateral :=
  (side1 side2 side3 side4 diagonal : ℝ)

/-- Checks if three lengths can form a valid triangle -/
def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Checks if the diagonal forms valid triangles with all possible combinations of sides -/
def diagonal_forms_valid_triangles (q : Quadrilateral) : Prop :=
  is_valid_triangle q.diagonal q.side1 q.side2 ∧
  is_valid_triangle q.diagonal q.side1 q.side3 ∧
  is_valid_triangle q.diagonal q.side1 q.side4 ∧
  is_valid_triangle q.diagonal q.side2 q.side3 ∧
  is_valid_triangle q.diagonal q.side2 q.side4 ∧
  is_valid_triangle q.diagonal q.side3 q.side4

theorem diagonal_length_is_2_8 (q : Quadrilateral) 
  (h1 : q.side1 = 1) (h2 : q.side2 = 2) (h3 : q.side3 = 5) (h4 : q.side4 = 7.5) (h5 : q.diagonal = 2.8) :
  diagonal_forms_valid_triangles q :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_is_2_8_l1807_180753


namespace NUMINAMATH_CALUDE_correct_selection_count_l1807_180745

/-- The number of ways to select representatives satisfying given conditions -/
def select_representatives (num_boys num_girls : ℕ) : ℕ :=
  let total_students := num_boys + num_girls
  let num_subjects := 5
  3360

/-- The theorem stating the correct number of ways to select representatives -/
theorem correct_selection_count :
  select_representatives 5 3 = 3360 := by sorry

end NUMINAMATH_CALUDE_correct_selection_count_l1807_180745


namespace NUMINAMATH_CALUDE_abc_xyz_inequality_l1807_180733

theorem abc_xyz_inequality (a b c x y z : ℝ) 
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a*x + b*y + c*z ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_xyz_inequality_l1807_180733


namespace NUMINAMATH_CALUDE_complement_connected_if_not_connected_l1807_180762

-- Define a graph
def Graph := Type

-- Define the property of being connected
def is_connected (G : Graph) : Prop := sorry

-- Define the complement of a graph
def complement (G : Graph) : Graph := sorry

-- Theorem statement
theorem complement_connected_if_not_connected (G : Graph) :
  ¬(is_connected G) → is_connected (complement G) := by sorry

end NUMINAMATH_CALUDE_complement_connected_if_not_connected_l1807_180762


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1807_180768

theorem max_value_of_expression (t : ℝ) :
  (∃ (c : ℝ), ∀ (t : ℝ), (3^t - 4*t)*t / 9^t ≤ c) ∧
  (∃ (t : ℝ), (3^t - 4*t)*t / 9^t = 1/16) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1807_180768


namespace NUMINAMATH_CALUDE_half_squared_equals_quarter_l1807_180774

theorem half_squared_equals_quarter : (1 / 2 : ℝ) ^ 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_half_squared_equals_quarter_l1807_180774


namespace NUMINAMATH_CALUDE_atlantic_charge_proof_l1807_180756

/-- The base rate for United Telephone in dollars -/
def united_base_rate : ℚ := 9

/-- The additional charge per minute for United Telephone in dollars -/
def united_per_minute : ℚ := 1/4

/-- The base rate for Atlantic Call in dollars -/
def atlantic_base_rate : ℚ := 12

/-- The number of minutes for which the bills are equal -/
def equal_minutes : ℕ := 60

/-- The additional charge per minute for Atlantic Call in dollars -/
def atlantic_per_minute : ℚ := 1/5

theorem atlantic_charge_proof :
  united_base_rate + united_per_minute * equal_minutes =
  atlantic_base_rate + atlantic_per_minute * equal_minutes :=
sorry

end NUMINAMATH_CALUDE_atlantic_charge_proof_l1807_180756


namespace NUMINAMATH_CALUDE_problem_statement_l1807_180724

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/x + y < 1/a + b) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/x + y ≥ 4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → x/y ≤ 1/4) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ x/y = 1/4) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 1/y = 1 → 1/2 * y - x ≥ Real.sqrt 2 - 1) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 1/y = 1 ∧ 1/2 * y - x = Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l1807_180724


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_l1807_180765

/-- Represents a shape created by joining unit cubes -/
structure CubeShape where
  /-- The number of unit cubes in the shape -/
  num_cubes : ℕ
  /-- The number of cubes surrounding the center cube -/
  surrounding_cubes : ℕ
  /-- Whether there's an additional cube on top -/
  has_top_cube : Bool

/-- Calculates the volume of the shape -/
def volume (shape : CubeShape) : ℕ := shape.num_cubes

/-- Calculates the surface area of the shape -/
def surface_area (shape : CubeShape) : ℕ :=
  shape.surrounding_cubes * 4 + (if shape.has_top_cube then 5 else 0)

/-- The specific shape described in the problem -/
def problem_shape : CubeShape :=
  { num_cubes := 8
  , surrounding_cubes := 6
  , has_top_cube := true }

theorem volume_surface_area_ratio :
  (volume problem_shape : ℚ) / (surface_area problem_shape : ℚ) = 8 / 29 := by sorry

end NUMINAMATH_CALUDE_volume_surface_area_ratio_l1807_180765


namespace NUMINAMATH_CALUDE_triploid_oyster_principle_is_chromosome_variation_l1807_180767

/-- Represents the principle underlying oyster cultivation methods -/
inductive CultivationPrinciple
  | GeneticMutation
  | ChromosomeNumberVariation
  | GeneRecombination
  | ChromosomeStructureVariation

/-- Represents the ploidy level of an oyster -/
inductive Ploidy
  | Diploid
  | Triploid

/-- Represents the state of a cell during oyster reproduction -/
structure CellState where
  chromosomeSets : ℕ
  polarBodyReleased : Bool

/-- Represents the cultivation method for oysters -/
structure CultivationMethod where
  chemicalTreatment : Bool
  preventPolarBodyRelease : Bool
  solveFleshQualityDecline : Bool

/-- The principle of triploid oyster cultivation -/
def triploidOysterPrinciple (method : CultivationMethod) : CultivationPrinciple :=
  sorry

/-- Theorem stating that the principle of triploid oyster cultivation
    is chromosome number variation -/
theorem triploid_oyster_principle_is_chromosome_variation
  (method : CultivationMethod)
  (h1 : method.chemicalTreatment = true)
  (h2 : method.preventPolarBodyRelease = true)
  (h3 : method.solveFleshQualityDecline = true) :
  triploidOysterPrinciple method = CultivationPrinciple.ChromosomeNumberVariation :=
  sorry

end NUMINAMATH_CALUDE_triploid_oyster_principle_is_chromosome_variation_l1807_180767


namespace NUMINAMATH_CALUDE_inequality_proof_l1807_180790

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1807_180790


namespace NUMINAMATH_CALUDE_no_integer_solution_2016_equation_l1807_180714

theorem no_integer_solution_2016_equation :
  ¬∃ (x y z : ℤ), (2016 : ℚ) = (x^2 + y^2 + z^2 : ℚ) / (x*y + y*z + z*x : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_solution_2016_equation_l1807_180714


namespace NUMINAMATH_CALUDE_constant_k_value_l1807_180718

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 12)*x - 8 = -(x - 2)*(x - 4) ↔ k = -18 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_k_value_l1807_180718


namespace NUMINAMATH_CALUDE_house_area_proof_l1807_180706

def house_painting_problem (price_per_sqft : ℝ) (total_cost : ℝ) : Prop :=
  price_per_sqft > 0 ∧ total_cost > 0 ∧ (total_cost / price_per_sqft = 88)

theorem house_area_proof :
  house_painting_problem 20 1760 :=
sorry

end NUMINAMATH_CALUDE_house_area_proof_l1807_180706


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1807_180728

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) : ℕ → ℚ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

def sum_arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem arithmetic_sequence_common_difference :
  ∀ (d : ℚ), 
    (arithmetic_sequence 1 d 0 = 1) →
    (sum_arithmetic_sequence 1 d 5 = 20) →
    d = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1807_180728


namespace NUMINAMATH_CALUDE_parabola_vertex_l1807_180770

/-- The vertex of the parabola y = -2x^2 + 3 is (0, 3) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -2 * x^2 + 3 → (0, 3) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1807_180770


namespace NUMINAMATH_CALUDE_passengers_gained_at_halfway_l1807_180789

theorem passengers_gained_at_halfway (num_cars : ℕ) (initial_people_per_car : ℕ) (total_people_at_end : ℕ) : 
  num_cars = 20 →
  initial_people_per_car = 3 →
  total_people_at_end = 80 →
  (total_people_at_end - num_cars * initial_people_per_car) / num_cars = 1 :=
by sorry

end NUMINAMATH_CALUDE_passengers_gained_at_halfway_l1807_180789


namespace NUMINAMATH_CALUDE_home_appliances_promotion_l1807_180783

theorem home_appliances_promotion (salespersons technicians : ℕ) 
  (h1 : salespersons = 5)
  (h2 : technicians = 4)
  (h3 : salespersons + technicians = 9) :
  (Nat.choose 9 3) - (Nat.choose 5 3) - (Nat.choose 4 3) = 70 := by
  sorry

end NUMINAMATH_CALUDE_home_appliances_promotion_l1807_180783


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1807_180739

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  Complex.im ((1 / (1 + i)) + i^3) = -3/2 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1807_180739


namespace NUMINAMATH_CALUDE_tan_cos_eq_sin_minus_m_sin_l1807_180734

theorem tan_cos_eq_sin_minus_m_sin (m : ℝ) : 
  Real.tan (π / 12) * Real.cos (5 * π / 12) = Real.sin (5 * π / 12) - m * Real.sin (π / 12) → 
  m = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_cos_eq_sin_minus_m_sin_l1807_180734


namespace NUMINAMATH_CALUDE_largest_product_sum_1976_l1807_180721

theorem largest_product_sum_1976 (n : ℕ) (factors : List ℕ) : 
  (factors.sum = 1976) →
  (factors.prod ≤ 2 * 3^658) :=
sorry

end NUMINAMATH_CALUDE_largest_product_sum_1976_l1807_180721


namespace NUMINAMATH_CALUDE_average_of_next_ten_l1807_180742

def consecutive_integers_average (c d : ℤ) : Prop :=
  (7 * d = c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) ∧
  (c > 0)

theorem average_of_next_ten (c d : ℤ) 
  (h : consecutive_integers_average c d) : 
  (((d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + 
    (d + 6) + (d + 7) + (d + 8) + (d + 9) + (d + 10)) / 10) = c + 9 :=
by sorry

end NUMINAMATH_CALUDE_average_of_next_ten_l1807_180742


namespace NUMINAMATH_CALUDE_time_to_eat_half_l1807_180710

/-- Represents the eating rate of a bird in terms of fraction of nuts eaten per hour -/
structure BirdRate where
  fraction : ℚ
  hours : ℚ

/-- Calculates the rate at which a bird eats nuts per hour -/
def eatRate (br : BirdRate) : ℚ :=
  br.fraction / br.hours

/-- Represents the rates of the three birds -/
structure BirdRates where
  crow : BirdRate
  sparrow : BirdRate
  parrot : BirdRate

/-- Calculates the combined eating rate of all three birds -/
def combinedRate (rates : BirdRates) : ℚ :=
  eatRate rates.crow + eatRate rates.sparrow + eatRate rates.parrot

/-- The main theorem stating the time taken to eat half the nuts -/
theorem time_to_eat_half (rates : BirdRates) 
  (h_crow : rates.crow = ⟨1/5, 4⟩) 
  (h_sparrow : rates.sparrow = ⟨1/3, 6⟩)
  (h_parrot : rates.parrot = ⟨1/4, 8⟩) : 
  (1/2) / combinedRate rates = 2880 / 788 := by
  sorry

end NUMINAMATH_CALUDE_time_to_eat_half_l1807_180710


namespace NUMINAMATH_CALUDE_inequality_proof_l1807_180772

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_eq : 1 + a + b + c = 2 * a * b * c) : 
  (a * b) / (1 + a + b) + (b * c) / (1 + b + c) + (c * a) / (1 + c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1807_180772


namespace NUMINAMATH_CALUDE_inequality_solution_l1807_180707

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ (4 < x ∧ x < 5) ∨ (6 < x ∧ x < 8) ∨ 10 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1807_180707


namespace NUMINAMATH_CALUDE_goldfish_count_l1807_180776

/-- The number of goldfish in the fish tank -/
def num_goldfish : ℕ := sorry

/-- The number of platyfish in the fish tank -/
def num_platyfish : ℕ := 10

/-- The number of red balls each goldfish plays with -/
def red_balls_per_goldfish : ℕ := 10

/-- The number of white balls each platyfish plays with -/
def white_balls_per_platyfish : ℕ := 5

/-- The total number of balls in the fish tank -/
def total_balls : ℕ := 80

theorem goldfish_count : num_goldfish = 3 := by
  have h1 : num_goldfish * red_balls_per_goldfish + num_platyfish * white_balls_per_platyfish = total_balls := sorry
  sorry

end NUMINAMATH_CALUDE_goldfish_count_l1807_180776


namespace NUMINAMATH_CALUDE_cricket_target_run_l1807_180738

/-- Given a cricket game with specific run rates, calculate the target run. -/
theorem cricket_target_run (total_overs : ℕ) (first_overs : ℕ) (remaining_overs : ℕ)
  (first_rate : ℝ) (remaining_rate : ℝ) :
  total_overs = first_overs + remaining_overs →
  first_overs = 10 →
  remaining_overs = 22 →
  first_rate = 3.2 →
  remaining_rate = 11.363636363636363 →
  ↑⌊(first_overs : ℝ) * first_rate + (remaining_overs : ℝ) * remaining_rate⌋ = 282 := by
  sorry

end NUMINAMATH_CALUDE_cricket_target_run_l1807_180738


namespace NUMINAMATH_CALUDE_factors_of_48_l1807_180754

/-- The number of distinct positive factors of 48 -/
def num_factors_48 : ℕ := sorry

/-- Theorem stating that the number of distinct positive factors of 48 is 10 -/
theorem factors_of_48 : num_factors_48 = 10 := by sorry

end NUMINAMATH_CALUDE_factors_of_48_l1807_180754


namespace NUMINAMATH_CALUDE_die_throw_probability_l1807_180764

/-- Represents a fair six-sided die throw -/
def DieFace := Fin 6

/-- The probability of a specific outcome when throwing a fair die three times -/
def prob_single_outcome : ℚ := 1 / 216

/-- Checks if a + bi is a root of x^2 - 2x + c = 0 -/
def is_root (a b c : ℕ) : Prop :=
  a = 1 ∧ c = b^2 + 1

/-- The number of favorable outcomes -/
def favorable_outcomes : ℕ := 2

theorem die_throw_probability :
  (favorable_outcomes : ℚ) * prob_single_outcome = 1 / 108 := by
  sorry


end NUMINAMATH_CALUDE_die_throw_probability_l1807_180764


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1807_180737

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to x-axis -/
def symmetric_x (A B : Point) : Prop :=
  B.x = A.x ∧ B.y = -A.y

theorem symmetric_point_coordinates :
  let A : Point := ⟨2, -1⟩
  ∀ B : Point, symmetric_x A B → B = ⟨2, 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1807_180737


namespace NUMINAMATH_CALUDE_square_exterior_points_distance_l1807_180701

/-- Given a square ABCD with side length 13 and exterior points E and F,
    prove that EF² = 578 when BE = DF = 5 and AE = CF = 12 -/
theorem square_exterior_points_distance (A B C D E F : ℝ × ℝ) : 
  let side_length : ℝ := 13
  -- Square ABCD
  A = (0, side_length) ∧ 
  B = (side_length, side_length) ∧ 
  C = (side_length, 0) ∧ 
  D = (0, 0) ∧
  -- Exterior points E and F
  dist B E = 5 ∧
  dist D F = 5 ∧
  dist A E = 12 ∧
  dist C F = 12
  →
  dist E F ^ 2 = 578 := by
sorry


end NUMINAMATH_CALUDE_square_exterior_points_distance_l1807_180701


namespace NUMINAMATH_CALUDE_positive_integers_divisibility_l1807_180746

theorem positive_integers_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_divisibility_l1807_180746


namespace NUMINAMATH_CALUDE_simplify_expression_l1807_180796

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1807_180796


namespace NUMINAMATH_CALUDE_mariam_neighborhood_houses_l1807_180778

/-- The number of houses on one side of the main road -/
def houses_on_first_side : ℕ := 40

/-- The function representing the number of houses on the other side of the road -/
def f (x : ℕ) : ℕ := x^2 + 3*x

/-- The total number of houses in Mariam's neighborhood -/
def total_houses : ℕ := houses_on_first_side + f houses_on_first_side

theorem mariam_neighborhood_houses :
  total_houses = 1760 := by sorry

end NUMINAMATH_CALUDE_mariam_neighborhood_houses_l1807_180778
