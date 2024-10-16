import Mathlib

namespace NUMINAMATH_CALUDE_video_votes_l4022_402290

theorem video_votes (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) : 
  likes + dislikes = total_votes →
  likes = (3 * total_votes) / 4 →
  dislikes = total_votes / 4 →
  likes - dislikes = 50 →
  total_votes = 100 := by
sorry

end NUMINAMATH_CALUDE_video_votes_l4022_402290


namespace NUMINAMATH_CALUDE_one_mile_in_yards_l4022_402227

-- Define the conversion rates
def mile_to_furlong : ℚ := 5
def furlong_to_rod : ℚ := 50
def rod_to_yard : ℚ := 5

-- Theorem statement
theorem one_mile_in_yards :
  mile_to_furlong * furlong_to_rod * rod_to_yard = 1250 := by
  sorry

end NUMINAMATH_CALUDE_one_mile_in_yards_l4022_402227


namespace NUMINAMATH_CALUDE_simplify_fraction_l4022_402257

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) : 
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l4022_402257


namespace NUMINAMATH_CALUDE_sin_330_degrees_l4022_402228

theorem sin_330_degrees : 
  Real.sin (330 * Real.pi / 180) = -Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l4022_402228


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l4022_402283

theorem arithmetic_evaluation : 8 * (6 - 4) + 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l4022_402283


namespace NUMINAMATH_CALUDE_sample_size_is_sixty_verify_conditions_l4022_402287

/-- Represents the total number of students in the population -/
def total_students : ℕ := 600

/-- Represents the number of male students in the population -/
def male_students : ℕ := 310

/-- Represents the number of female students in the population -/
def female_students : ℕ := 290

/-- Represents the number of male students in the sample -/
def sample_males : ℕ := 31

/-- Calculates the sample size based on stratified random sampling by gender -/
def calculate_sample_size (total : ℕ) (males : ℕ) (sample_males : ℕ) : ℕ :=
  (sample_males * total) / males

/-- Theorem stating that the calculated sample size is 60 -/
theorem sample_size_is_sixty :
  calculate_sample_size total_students male_students sample_males = 60 := by
  sorry

/-- Theorem verifying the given conditions -/
theorem verify_conditions :
  total_students = male_students + female_students ∧
  male_students = 310 ∧
  female_students = 290 ∧
  sample_males = 31 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_is_sixty_verify_conditions_l4022_402287


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l4022_402219

theorem ceiling_floor_difference : 
  ⌈(10 : ℝ) / 4 * (-17 : ℝ) / 2⌉ - ⌊(10 : ℝ) / 4 * ⌊(-17 : ℝ) / 2⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l4022_402219


namespace NUMINAMATH_CALUDE_oldest_babysat_age_l4022_402226

theorem oldest_babysat_age (jane_start_age : ℕ) (jane_current_age : ℕ) (years_since_stopped : ℕ) :
  jane_start_age = 18 →
  jane_current_age = 34 →
  years_since_stopped = 12 →
  (∀ (jane_age : ℕ) (child_age : ℕ),
    jane_age ≥ jane_start_age →
    jane_age ≤ jane_current_age - years_since_stopped →
    child_age ≤ jane_age / 2) →
  (jane_current_age - years_since_stopped - jane_start_age) + years_since_stopped + 
    ((jane_current_age - years_since_stopped) / 2) = 23 :=
by sorry

end NUMINAMATH_CALUDE_oldest_babysat_age_l4022_402226


namespace NUMINAMATH_CALUDE_parabola_above_line_l4022_402242

theorem parabola_above_line (p : ℝ) : 
  (∀ x : ℝ, x^2 - 2*p*x + p + 1 ≥ -12*x + 5) ↔ (5 ≤ p ∧ p ≤ 8) := by
  sorry

end NUMINAMATH_CALUDE_parabola_above_line_l4022_402242


namespace NUMINAMATH_CALUDE_tan_sum_x_y_pi_third_l4022_402249

theorem tan_sum_x_y_pi_third (x y m : ℝ) 
  (hx : x^3 + Real.sin (2*x) = m)
  (hy : y^3 + Real.sin (2*y) = -m)
  (hx_bound : x > -π/4 ∧ x < π/4)
  (hy_bound : y > -π/4 ∧ y < π/4) :
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_x_y_pi_third_l4022_402249


namespace NUMINAMATH_CALUDE_complement_of_A_union_B_l4022_402208

def U : Set Int := {-1, 0, 1, 2, 3, 4}
def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2, 3}

theorem complement_of_A_union_B : (A ∪ B)ᶜ = {4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_union_B_l4022_402208


namespace NUMINAMATH_CALUDE_jerome_classmates_l4022_402210

/-- Represents Jerome's contact list --/
structure ContactList where
  classmates : ℕ
  outOfSchoolFriends : ℕ
  familyMembers : ℕ
  total : ℕ

/-- The properties of Jerome's contact list --/
def jeromeContactList : ContactList → Prop
  | cl => cl.outOfSchoolFriends = cl.classmates / 2 ∧
          cl.familyMembers = 3 ∧
          cl.total = 33 ∧
          cl.total = cl.classmates + cl.outOfSchoolFriends + cl.familyMembers

/-- Theorem: Jerome has 20 classmates on his contact list --/
theorem jerome_classmates :
  ∀ cl : ContactList, jeromeContactList cl → cl.classmates = 20 := by
  sorry


end NUMINAMATH_CALUDE_jerome_classmates_l4022_402210


namespace NUMINAMATH_CALUDE_number_percentage_problem_l4022_402223

theorem number_percentage_problem (N : ℚ) : 
  (4/5 : ℚ) * (3/8 : ℚ) * N = 24 → (5/2 : ℚ) * N = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l4022_402223


namespace NUMINAMATH_CALUDE_root_equation_value_l4022_402214

theorem root_equation_value (a : ℝ) : 
  a^2 + 3*a - 1 = 0 → 2*a^2 + 6*a + 2021 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l4022_402214


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l4022_402246

theorem salary_increase_percentage (S : ℝ) (x : ℝ) 
  (h1 : S + 0.15 * S = 575) 
  (h2 : S + x * S = 600) : 
  x = 0.2 := by
sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l4022_402246


namespace NUMINAMATH_CALUDE_chocolates_remaining_chocolates_remaining_day6_l4022_402255

/-- Chocolates remaining after 5 days of eating with given conditions -/
theorem chocolates_remaining (total : ℕ) (day1 : ℕ) (day2 : ℕ) : ℕ :=
  let day3 := day1 - 3
  let day4 := 2 * day3 + 1
  let day5 := day2 / 2
  total - (day1 + day2 + day3 + day4 + day5)

/-- Proof that 14 chocolates remain on Day 6 given the problem conditions -/
theorem chocolates_remaining_day6 :
  chocolates_remaining 48 6 8 = 14 := by
  sorry

end NUMINAMATH_CALUDE_chocolates_remaining_chocolates_remaining_day6_l4022_402255


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l4022_402217

/-- Represents a number in a given base with two identical digits --/
def twoDigitNumber (digit : Nat) (base : Nat) : Nat :=
  digit * base + digit

/-- Checks if a digit is valid in a given base --/
def isValidDigit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    isValidDigit C 4 ∧
    isValidDigit D 6 ∧
    twoDigitNumber C 4 = 35 ∧
    twoDigitNumber D 6 = 35 ∧
    (∀ (C' D' : Nat),
      isValidDigit C' 4 →
      isValidDigit D' 6 →
      twoDigitNumber C' 4 = twoDigitNumber D' 6 →
      twoDigitNumber C' 4 ≥ 35) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l4022_402217


namespace NUMINAMATH_CALUDE_infinite_triples_existence_l4022_402261

theorem infinite_triples_existence :
  ∀ m : ℕ, ∃ p : ℕ, ∃ q₁ q₂ : ℤ,
    p > m ∧ 
    |Real.sqrt 2 - (q₁ : ℝ) / p| * |Real.sqrt 3 - (q₂ : ℝ) / p| ≤ 1 / (2 * (p : ℝ) ^ 3) :=
by sorry

end NUMINAMATH_CALUDE_infinite_triples_existence_l4022_402261


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l4022_402200

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l4022_402200


namespace NUMINAMATH_CALUDE_remainder_2_pow_1984_mod_17_l4022_402267

theorem remainder_2_pow_1984_mod_17 (h1 : Prime 17) (h2 : ¬ 17 ∣ 2) :
  2^1984 ≡ 0 [ZMOD 17] := by
  sorry

end NUMINAMATH_CALUDE_remainder_2_pow_1984_mod_17_l4022_402267


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l4022_402282

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 3 * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value (a : ℝ) :
  (a > 0) →
  (∃ x y : ℝ, hyperbola_equation x y a ∧ asymptote_equation x y) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l4022_402282


namespace NUMINAMATH_CALUDE_class_size_l4022_402248

theorem class_size (hindi : ℕ) (english : ℕ) (both : ℕ) (total : ℕ) : 
  hindi = 30 → 
  english = 20 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  total = 40 := by
sorry

end NUMINAMATH_CALUDE_class_size_l4022_402248


namespace NUMINAMATH_CALUDE_expense_calculation_correct_l4022_402202

/-- Calculates the total out-of-pocket expense for James' purchases and transactions --/
def total_expense (initial_purchase : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
  (tv_cost : ℝ) (bike_cost : ℝ) (usd_to_eur_initial : ℝ) (usd_to_eur_refund : ℝ)
  (usd_to_gbp : ℝ) (other_bike_markup : ℝ) (other_bike_sale_rate : ℝ)
  (toaster_cost_eur : ℝ) (microwave_cost_eur : ℝ)
  (subscription_cost_gbp : ℝ) (subscription_discount : ℝ) (subscription_months : ℕ) : ℝ :=
  sorry

/-- The total out-of-pocket expense matches the calculated value --/
theorem expense_calculation_correct :
  total_expense 5000 0.1 0.05 1000 700 0.85 0.87 0.77 0.2 0.85 100 150 80 0.3 12 = 2291.63 :=
  sorry

end NUMINAMATH_CALUDE_expense_calculation_correct_l4022_402202


namespace NUMINAMATH_CALUDE_class_size_l4022_402231

/-- Represents the number of students in the class -/
def n : ℕ := sorry

/-- Represents the number of leftover erasers -/
def x : ℕ := sorry

/-- The number of gel pens bought -/
def gel_pens : ℕ := 2 * n + 2 * x

/-- The number of ballpoint pens bought -/
def ballpoint_pens : ℕ := 3 * n + 48

/-- The number of erasers bought -/
def erasers : ℕ := 4 * n + x

theorem class_size : 
  gel_pens = ballpoint_pens ∧ 
  ballpoint_pens = erasers ∧ 
  x = 2 * n → 
  n = 16 := by sorry

end NUMINAMATH_CALUDE_class_size_l4022_402231


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4022_402233

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 4 = 3) ∧ 
  (x % 6 = 5) ∧ 
  (∀ (y : ℕ), y > 0 → y % 4 = 3 → y % 6 = 5 → x ≤ y) ∧
  (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l4022_402233


namespace NUMINAMATH_CALUDE_parallelogram_area_l4022_402235

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 12 inches and 20 inches is equal to 120 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 12 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin θ = 120 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l4022_402235


namespace NUMINAMATH_CALUDE_max_product_on_curve_l4022_402201

theorem max_product_on_curve (x y : ℝ) :
  0 ≤ x ∧ x ≤ 12 ∧ 0 ≤ y ∧ y ≤ 12 →
  x * y = (12 - x)^2 * (12 - y)^2 →
  x * y ≤ 81 :=
by sorry

end NUMINAMATH_CALUDE_max_product_on_curve_l4022_402201


namespace NUMINAMATH_CALUDE_budget_allocation_l4022_402297

theorem budget_allocation (microphotonics home_electronics food_additives industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 ∧
  home_electronics = 19 ∧
  food_additives = 10 ∧
  industrial_lubricants = 8 ∧
  basic_astrophysics_degrees = 90 →
  ∃ (genetically_modified_microorganisms : ℝ),
    genetically_modified_microorganisms = 24 ∧
    microphotonics + home_electronics + food_additives + industrial_lubricants +
    (basic_astrophysics_degrees / 360 * 100) + genetically_modified_microorganisms = 100 :=
by sorry

end NUMINAMATH_CALUDE_budget_allocation_l4022_402297


namespace NUMINAMATH_CALUDE_fraction_simplification_l4022_402237

theorem fraction_simplification (x y : ℚ) (hx : x = 2) (hy : y = 3) :
  (1 / y) / (1 / x) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4022_402237


namespace NUMINAMATH_CALUDE_sum_equals_210_l4022_402276

theorem sum_equals_210 : 145 + 35 + 25 + 5 = 210 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_210_l4022_402276


namespace NUMINAMATH_CALUDE_incorrect_statement_B_l4022_402275

/-- Definition of a "2 times root equation" -/
def is_two_times_root_equation (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ (a*x^2 + b*x + c = 0) ∧ (a*y^2 + b*y + c = 0) ∧ (x = 2*y ∨ y = 2*x)

/-- The statement to be proven false -/
theorem incorrect_statement_B :
  ¬(∀ (m n : ℝ), is_two_times_root_equation 1 (m-2) (-2*m) → m + n = 0) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_B_l4022_402275


namespace NUMINAMATH_CALUDE_ellipse_properties_l4022_402230

/-- An ellipse with center at the origin, foci on the x-axis, and max/min distances to focus 3 and 1 -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  max_dist : ℝ := 3
  min_dist : ℝ := 1

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- A line with equation y = x + m -/
structure Line where
  m : ℝ

/-- Predicate for line intersection with ellipse -/
def intersects (l : Line) (e : Ellipse) : Prop :=
  ∃ x y : ℝ, y = x + l.m ∧ standard_equation e x y

theorem ellipse_properties (e : Ellipse) :
  (∀ x y : ℝ, standard_equation e x y ↔ 
    x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ l : Line, intersects l e ↔ 
    -Real.sqrt 7 ≤ l.m ∧ l.m ≤ Real.sqrt 7) := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4022_402230


namespace NUMINAMATH_CALUDE_range_of_expression_l4022_402234

theorem range_of_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  0 < x * y + y * z + z * x - 2 * x * y * z ∧ 
  x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l4022_402234


namespace NUMINAMATH_CALUDE_odd_products_fraction_l4022_402292

def multiplication_table_size : ℕ := 11

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_numbers (n : ℕ) : ℕ := (n + 1) / 2

theorem odd_products_fraction :
  (count_odd_numbers multiplication_table_size)^2 / multiplication_table_size^2 = 25 / 121 := by
  sorry

end NUMINAMATH_CALUDE_odd_products_fraction_l4022_402292


namespace NUMINAMATH_CALUDE_oliver_card_collection_l4022_402254

theorem oliver_card_collection (monster_club : ℕ) (alien_baseball : ℕ) (battle_gremlins : ℕ) 
  (h1 : monster_club = 2 * alien_baseball)
  (h2 : battle_gremlins = 48)
  (h3 : battle_gremlins = 3 * alien_baseball) :
  monster_club = 32 := by
  sorry

end NUMINAMATH_CALUDE_oliver_card_collection_l4022_402254


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l4022_402259

theorem floor_of_expression_equals_eight :
  ⌊(2005^3 : ℝ) / (2003 * 2004) - (2003^3 : ℝ) / (2004 * 2005)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l4022_402259


namespace NUMINAMATH_CALUDE_dot_product_range_l4022_402236

theorem dot_product_range (a b : EuclideanSpace ℝ (Fin n)) :
  norm a = 2 →
  norm b = 2 →
  (∀ x : ℝ, norm (a + x • b) ≥ 1) →
  -2 * Real.sqrt 3 ≤ inner a b ∧ inner a b ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_range_l4022_402236


namespace NUMINAMATH_CALUDE_all_offers_count_l4022_402293

def stadium_capacity : ℕ := 4500

def hot_dog_interval : ℕ := 90
def soda_interval : ℕ := 45
def popcorn_interval : ℕ := 60
def ice_cream_interval : ℕ := 45

def fans_with_all_offers : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval popcorn_interval))

theorem all_offers_count :
  fans_with_all_offers = 25 :=
sorry

end NUMINAMATH_CALUDE_all_offers_count_l4022_402293


namespace NUMINAMATH_CALUDE_incorrect_permutations_of_good_l4022_402218

-- Define a structure for our word
structure Word where
  length : Nat
  repeated_letter_count : Nat

-- Define our specific word "good"
def good : Word := { length := 4, repeated_letter_count := 2 }

-- Theorem statement
theorem incorrect_permutations_of_good (w : Word) (h1 : w = good) : 
  (w.length.factorial / w.repeated_letter_count.factorial) - 1 = 11 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_permutations_of_good_l4022_402218


namespace NUMINAMATH_CALUDE_marble_ratio_l4022_402207

theorem marble_ratio : 
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := 4
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  (dark_blue_marbles : ℚ) / total_marbles = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_ratio_l4022_402207


namespace NUMINAMATH_CALUDE_tax_rate_change_income_l4022_402260

/-- Proves that given the conditions of the tax rate change and differential savings, 
    the taxpayer's annual income before tax is $45,000 -/
theorem tax_rate_change_income (I : ℝ) 
  (h1 : 0.40 * I - 0.33 * I = 3150) : I = 45000 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_change_income_l4022_402260


namespace NUMINAMATH_CALUDE_reciprocal_problem_l4022_402211

theorem reciprocal_problem (x : ℚ) (h : 7 * x = 3) : 150 * (1 / x) = 350 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l4022_402211


namespace NUMINAMATH_CALUDE_line_translation_theorem_l4022_402264

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line vertically -/
def translate_line (l : Line) (units : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept + units }

theorem line_translation_theorem (original : Line) (units : ℝ) :
  original.slope = 1/2 ∧ original.intercept = -2 ∧ units = 3 →
  translate_line original units = Line.mk (1/2) 1 :=
by sorry

end NUMINAMATH_CALUDE_line_translation_theorem_l4022_402264


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4022_402299

/-- A quadratic function f(x) = x² + bx - 5 with axis of symmetry at x = 2 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 5

/-- The axis of symmetry of f is at x = 2 -/
def axis_of_symmetry (b : ℝ) : Prop := -b/(2*1) = 2

/-- The equation f(x) = 2x - 13 -/
def equation (b : ℝ) (x : ℝ) : Prop := f b x = 2*x - 13

theorem quadratic_equation_solution (b : ℝ) :
  axis_of_symmetry b →
  (∃ x y : ℝ, x = 2 ∧ y = 4 ∧ equation b x ∧ equation b y ∧
    ∀ z : ℝ, equation b z → (z = x ∨ z = y)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4022_402299


namespace NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l4022_402256

theorem units_digit_of_n_squared_plus_two_to_n (n : ℕ) :
  n = 2023^2 + 2^2023 →
  (n^2 + 2^n) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_squared_plus_two_to_n_l4022_402256


namespace NUMINAMATH_CALUDE_inequality_solution_l4022_402266

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5) ↔ (x ≤ -8 ∨ (-2 ≤ x ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4022_402266


namespace NUMINAMATH_CALUDE_product_mod_six_l4022_402224

theorem product_mod_six : (2015 * 2016 * 2017 * 2018) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_six_l4022_402224


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l4022_402252

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_l4022_402252


namespace NUMINAMATH_CALUDE_john_smith_payment_l4022_402215

-- Define the number of cakes
def num_cakes : ℕ := 3

-- Define the cost per cake in cents (to avoid floating-point numbers)
def cost_per_cake : ℕ := 1200

-- Define the number of people sharing the cost
def num_people : ℕ := 2

-- Theorem to prove
theorem john_smith_payment :
  (num_cakes * cost_per_cake) / num_people = 1800 := by
  sorry

end NUMINAMATH_CALUDE_john_smith_payment_l4022_402215


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l4022_402204

/-- Given a quadratic equation (a-1)x² + x + a² - 1 = 0 where one root is 0, prove that a = -1 -/
theorem quadratic_root_zero (a : ℝ) : 
  (∀ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ↔ x = 0 ∨ x ≠ 0) →
  (∃ x, (a - 1) * x^2 + x + a^2 - 1 = 0 ∧ x = 0) →
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l4022_402204


namespace NUMINAMATH_CALUDE_boxes_with_neither_l4022_402241

theorem boxes_with_neither (total : ℕ) (stickers : ℕ) (stamps : ℕ) (both : ℕ) 
  (h1 : total = 15)
  (h2 : stickers = 9)
  (h3 : stamps = 5)
  (h4 : both = 3) : 
  total - (stickers + stamps - both) = 4 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l4022_402241


namespace NUMINAMATH_CALUDE_odd_divisors_of_square_plus_one_l4022_402278

theorem odd_divisors_of_square_plus_one (x : ℤ) (d : ℤ) (h : d ∣ x^2 + 1) (hodd : Odd d) :
  ∃ (k : ℤ), d = 4 * k + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_divisors_of_square_plus_one_l4022_402278


namespace NUMINAMATH_CALUDE_exam_mean_score_l4022_402289

/-- Given a distribution where 60 is 2 standard deviations below the mean
    and 100 is 3 standard deviations above the mean, the mean of the
    distribution is 76. -/
theorem exam_mean_score (μ σ : ℝ)
    (below_mean : μ - 2 * σ = 60)
    (above_mean : μ + 3 * σ = 100) :
    μ = 76 := by
  sorry


end NUMINAMATH_CALUDE_exam_mean_score_l4022_402289


namespace NUMINAMATH_CALUDE_inequality_proof_l4022_402294

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 4) 
  (h2 : c^2 + d^2 = 16) : 
  a*c + b*d ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l4022_402294


namespace NUMINAMATH_CALUDE_honor_students_count_l4022_402229

theorem honor_students_count 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (honor_girls : ℕ) 
  (honor_boys : ℕ) : 
  total_students < 30 →
  total_students = girls + boys →
  (honor_girls : ℚ) / girls = 3 / 13 →
  (honor_boys : ℚ) / boys = 4 / 11 →
  honor_girls + honor_boys = 7 :=
by sorry

end NUMINAMATH_CALUDE_honor_students_count_l4022_402229


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l4022_402243

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l4022_402243


namespace NUMINAMATH_CALUDE_proportion_solution_l4022_402281

theorem proportion_solution (x : ℚ) : (3/4 : ℚ) / x = 7/8 → x = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l4022_402281


namespace NUMINAMATH_CALUDE_incorrect_parentheses_removal_l4022_402232

theorem incorrect_parentheses_removal (a b c : ℝ) : c - 2*(a + b) ≠ c - 2*a + 2*b := by
  sorry

end NUMINAMATH_CALUDE_incorrect_parentheses_removal_l4022_402232


namespace NUMINAMATH_CALUDE_emily_caught_four_trout_l4022_402253

def fishing_problem (num_trout : ℕ) : Prop :=
  let num_catfish : ℕ := 3
  let num_bluegill : ℕ := 5
  let weight_trout : ℝ := 2
  let weight_catfish : ℝ := 1.5
  let weight_bluegill : ℝ := 2.5
  let total_weight : ℝ := 25
  (num_trout : ℝ) * weight_trout + 
  (num_catfish : ℝ) * weight_catfish + 
  (num_bluegill : ℝ) * weight_bluegill = total_weight

theorem emily_caught_four_trout : 
  ∃ (n : ℕ), fishing_problem n ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_caught_four_trout_l4022_402253


namespace NUMINAMATH_CALUDE_tom_filled_33_balloons_l4022_402220

/-- The number of water balloons filled up by Anthony -/
def anthony_balloons : ℕ := 44

/-- The number of water balloons filled up by Luke -/
def luke_balloons : ℕ := anthony_balloons / 4

/-- The number of water balloons filled up by Tom -/
def tom_balloons : ℕ := 3 * luke_balloons

/-- Theorem stating that Tom filled up 33 water balloons -/
theorem tom_filled_33_balloons : tom_balloons = 33 := by
  sorry

end NUMINAMATH_CALUDE_tom_filled_33_balloons_l4022_402220


namespace NUMINAMATH_CALUDE_stratified_sampling_results_l4022_402205

theorem stratified_sampling_results (total_sample : ℕ) (junior_pop : ℕ) (senior_pop : ℕ)
  (h1 : total_sample = 60)
  (h2 : junior_pop = 400)
  (h3 : senior_pop = 200) :
  (Nat.choose junior_pop ((junior_pop * total_sample) / (junior_pop + senior_pop))) *
  (Nat.choose senior_pop ((senior_pop * total_sample) / (junior_pop + senior_pop))) =
  (Nat.choose 400 40) * (Nat.choose 200 20) :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l4022_402205


namespace NUMINAMATH_CALUDE_hash_three_eight_equals_eighty_l4022_402251

-- Define the operation #
def hash (a b : ℝ) : ℝ := a * b - b + b^2

-- Theorem statement
theorem hash_three_eight_equals_eighty : hash 3 8 = 80 := by
  sorry

end NUMINAMATH_CALUDE_hash_three_eight_equals_eighty_l4022_402251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l4022_402212

/-- An arithmetic sequence with the given properties -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_a2 : a 2 = 12)
  (h_an : ∃ n : ℕ, a n = -20)
  (h_d : ∃ d : ℤ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∃ n : ℕ, n = 18 ∧ a n = -20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l4022_402212


namespace NUMINAMATH_CALUDE_candy_sharing_l4022_402272

theorem candy_sharing (bags : ℕ) (candies_per_bag : ℕ) (people : ℕ) :
  bags = 25 →
  candies_per_bag = 16 →
  people = 2 →
  (bags * candies_per_bag) / people = 200 := by
  sorry

end NUMINAMATH_CALUDE_candy_sharing_l4022_402272


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4022_402263

theorem fraction_to_decimal : (67 : ℚ) / (2^3 * 5^4) = 0.0134 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4022_402263


namespace NUMINAMATH_CALUDE_double_reflection_of_H_l4022_402245

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_line (p : ℝ × ℝ) : ℝ × ℝ := (p.2 + 1, p.1 + 1)

def H : ℝ × ℝ := (5, 1)

theorem double_reflection_of_H :
  reflect_line (reflect_x H) = (0, 4) := by sorry

end NUMINAMATH_CALUDE_double_reflection_of_H_l4022_402245


namespace NUMINAMATH_CALUDE_no_real_solutions_for_complex_norm_equation_l4022_402284

theorem no_real_solutions_for_complex_norm_equation :
  ¬∃ c : ℝ, Complex.abs (1 + c - 3*I) = 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_complex_norm_equation_l4022_402284


namespace NUMINAMATH_CALUDE_average_of_polynomials_l4022_402273

theorem average_of_polynomials (x : ℚ) : 
  (1 / 3 : ℚ) * ((x^2 - 3*x + 2) + (3*x^2 + x - 1) + (2*x^2 - 5*x + 7)) = 2*x^2 + 4 →
  x = -4/7 := by
sorry

end NUMINAMATH_CALUDE_average_of_polynomials_l4022_402273


namespace NUMINAMATH_CALUDE_transformed_system_solution_l4022_402239

/-- Given a system of equations with solution, prove that a transformed system has a specific solution -/
theorem transformed_system_solution (a b m n : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 10 ∧ m * x - n * y = 8 ∧ x = 1 ∧ y = 2) →
  (∃ x y : ℝ, (1/2) * a * (x + y) + (1/3) * b * (x - y) = 10 ∧
              (1/2) * m * (x + y) - (1/3) * n * (x - y) = 8 ∧
              x = 4 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_transformed_system_solution_l4022_402239


namespace NUMINAMATH_CALUDE_calculate_expression_l4022_402250

theorem calculate_expression : 
  4 * Real.sin (60 * π / 180) + (-1/3)⁻¹ - Real.sqrt 12 + abs (-5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4022_402250


namespace NUMINAMATH_CALUDE_total_students_present_l4022_402222

/-- Represents a kindergarten session with registered and absent students -/
structure Session where
  registered : ℕ
  absent : ℕ

/-- Calculates the number of present students in a session -/
def presentStudents (s : Session) : ℕ := s.registered - s.absent

/-- Represents the kindergarten school data -/
structure KindergartenSchool where
  morning : Session
  earlyAfternoon : Session
  lateAfternoon : Session
  earlyEvening : Session
  lateEvening : Session
  transferredOut : ℕ
  newRegistrations : ℕ
  newAttending : ℕ

/-- The main theorem to prove -/
theorem total_students_present (school : KindergartenSchool)
  (h1 : school.morning = { registered := 75, absent := 9 })
  (h2 : school.earlyAfternoon = { registered := 72, absent := 12 })
  (h3 : school.lateAfternoon = { registered := 90, absent := 15 })
  (h4 : school.earlyEvening = { registered := 50, absent := 6 })
  (h5 : school.lateEvening = { registered := 60, absent := 10 })
  (h6 : school.transferredOut = 3)
  (h7 : school.newRegistrations = 3)
  (h8 : school.newAttending = 1) :
  presentStudents school.morning +
  presentStudents school.earlyAfternoon +
  presentStudents school.lateAfternoon +
  presentStudents school.earlyEvening +
  presentStudents school.lateEvening -
  school.transferredOut +
  school.newAttending = 293 := by
  sorry

end NUMINAMATH_CALUDE_total_students_present_l4022_402222


namespace NUMINAMATH_CALUDE_jessica_quarters_l4022_402295

theorem jessica_quarters (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 :=
by sorry

end NUMINAMATH_CALUDE_jessica_quarters_l4022_402295


namespace NUMINAMATH_CALUDE_inequality_proof_l4022_402288

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) :
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4022_402288


namespace NUMINAMATH_CALUDE_k_range_oa_perpendicular_ob_l4022_402279

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points A and B
def intersection_points (k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2)

-- Define the vertex O of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Theorem for the range of k
theorem k_range : 
  ∀ k : ℝ, intersection_points k ↔ k ≠ 0 :=
sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

-- Theorem for perpendicularity of OA and OB
theorem oa_perpendicular_ob (k : ℝ) :
  k ≠ 0 → 
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola x1 y1 ∧ line k x1 y1 ∧
    parabola x2 y2 ∧ line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    perpendicular vertex (x1, y1) (x2, y2) :=
sorry

end NUMINAMATH_CALUDE_k_range_oa_perpendicular_ob_l4022_402279


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l4022_402280

theorem sin_cos_sum_equals_quarter :
  Real.sin (20 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_quarter_l4022_402280


namespace NUMINAMATH_CALUDE_twelve_returning_sequences_l4022_402258

-- Define the triangle T'
structure Triangle :=
  (v1 v2 v3 : ℝ × ℝ)

-- Define the set of transformations
inductive Transformation
  | Rotate60 : Transformation
  | Rotate120 : Transformation
  | Rotate240 : Transformation
  | ReflectYeqX : Transformation
  | ReflectYeqNegX : Transformation

-- Define a sequence of three transformations
def TransformationSequence := (Transformation × Transformation × Transformation)

-- Define the original triangle T'
def T' : Triangle :=
  { v1 := (1, 1), v2 := (5, 1), v3 := (1, 4) }

-- Function to check if a sequence of transformations returns T' to its original position
def returnsToOriginal (seq : TransformationSequence) : Prop :=
  sorry

-- Theorem stating that exactly 12 sequences return T' to its original position
theorem twelve_returning_sequences :
  ∃ (S : Finset TransformationSequence),
    (∀ seq ∈ S, returnsToOriginal seq) ∧
    (∀ seq, returnsToOriginal seq → seq ∈ S) ∧
    Finset.card S = 12 :=
  sorry

end NUMINAMATH_CALUDE_twelve_returning_sequences_l4022_402258


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l4022_402265

theorem stratified_sampling_sample_size 
  (total_teachers : ℕ) 
  (total_male_students : ℕ) 
  (total_female_students : ℕ) 
  (sampled_female_students : ℕ) 
  (h1 : total_teachers = 100) 
  (h2 : total_male_students = 600) 
  (h3 : total_female_students = 500) 
  (h4 : sampled_female_students = 40) : 
  (sampled_female_students : ℚ) / total_female_students = 
  96 / (total_teachers + total_male_students + total_female_students) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l4022_402265


namespace NUMINAMATH_CALUDE_zeros_before_last_digit_2009_pow_2011_l4022_402225

theorem zeros_before_last_digit_2009_pow_2011 :
  ∃ n : ℕ, n > 0 ∧ (2009^2011 % 10^(n+1)) / 10^n = 0 ∧ (2009^2011 % 10^n) / 10^(n-1) ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_zeros_before_last_digit_2009_pow_2011_l4022_402225


namespace NUMINAMATH_CALUDE_part1_part2_l4022_402216

-- Part 1
theorem part1 (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (hab : a + b = 10) (hxy : a / x + b / y = 1) (hmin : ∀ x' y', x' > 0 → y' > 0 → a / x' + b / y' = 1 → x' + y' ≥ 18) :
  (a = 2 ∧ b = 8) ∨ (a = 8 ∧ b = 2) := by sorry

-- Part 2
theorem part2 :
  ∃ a : ℝ, a > 0 ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a * (x + y)) ∧
  (∀ a' : ℝ, a' > 0 → (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * Real.sqrt (2 * x * y) ≤ a' * (x + y)) → a ≤ a') ∧
  a = 2 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l4022_402216


namespace NUMINAMATH_CALUDE_circular_garden_area_increase_l4022_402203

theorem circular_garden_area_increase : 
  let original_diameter : ℝ := 20
  let new_diameter : ℝ := 30
  let original_area := π * (original_diameter / 2)^2
  let new_area := π * (new_diameter / 2)^2
  let area_increase := new_area - original_area
  let percent_increase := (area_increase / original_area) * 100
  percent_increase = 125 := by sorry

end NUMINAMATH_CALUDE_circular_garden_area_increase_l4022_402203


namespace NUMINAMATH_CALUDE_walter_zoo_time_l4022_402240

theorem walter_zoo_time (S : ℝ) : 
  S > 0 ∧ 
  S + 8*S + 13 + S/2 = 185 → 
  S = 16 :=
by sorry

end NUMINAMATH_CALUDE_walter_zoo_time_l4022_402240


namespace NUMINAMATH_CALUDE_jordan_machine_input_l4022_402221

theorem jordan_machine_input (x : ℝ) : 2 * x + 3 - 5 = 27 → x = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_jordan_machine_input_l4022_402221


namespace NUMINAMATH_CALUDE_course_selection_theorem_l4022_402285

/-- The number of ways to select 4 courses out of 9, where 3 specific courses are mutually exclusive -/
def course_selection_schemes (total_courses : ℕ) (exclusive_courses : ℕ) (courses_to_choose : ℕ) : ℕ :=
  (exclusive_courses * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  Nat.choose (total_courses - exclusive_courses) courses_to_choose

/-- Theorem stating that the number of course selection schemes is 75 -/
theorem course_selection_theorem : course_selection_schemes 9 3 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l4022_402285


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4022_402206

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 + a 2 + a 3 = 7) →
  (a 2 + a 3 + a 4 = 14) →
  (a 4 + a 5 + a 6 = 56) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4022_402206


namespace NUMINAMATH_CALUDE_no_functions_exist_for_part_a_functions_exist_for_part_b_l4022_402270

-- Part (a)
theorem no_functions_exist_for_part_a :
  ¬∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3 := by sorry

-- Part (b)
theorem functions_exist_for_part_b :
  ∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^4 := by sorry

end NUMINAMATH_CALUDE_no_functions_exist_for_part_a_functions_exist_for_part_b_l4022_402270


namespace NUMINAMATH_CALUDE_unique_prime_arith_seq_l4022_402271

/-- An arithmetic sequence of three prime numbers with common difference 80. -/
structure PrimeArithSeq where
  p₁ : ℕ
  p₂ : ℕ
  p₃ : ℕ
  prime_p₁ : Nat.Prime p₁
  prime_p₂ : Nat.Prime p₂
  prime_p₃ : Nat.Prime p₃
  diff_p₂_p₁ : p₂ = p₁ + 80
  diff_p₃_p₂ : p₃ = p₂ + 80

/-- There exists exactly one arithmetic sequence of three prime numbers with common difference 80. -/
theorem unique_prime_arith_seq : ∃! seq : PrimeArithSeq, True :=
  sorry

end NUMINAMATH_CALUDE_unique_prime_arith_seq_l4022_402271


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l4022_402238

theorem dogwood_trees_planted_tomorrow 
  (initial_trees : ℕ) 
  (planted_today : ℕ) 
  (final_total : ℕ) :
  initial_trees = 7 →
  planted_today = 3 →
  final_total = 12 →
  final_total - (initial_trees + planted_today) = 2 :=
by sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l4022_402238


namespace NUMINAMATH_CALUDE_toms_seashells_l4022_402213

/-- Calculates the number of unbroken seashells Tom had left after three days of collecting and giving some away. -/
theorem toms_seashells (day1_total day1_broken day2_total day2_broken day3_total day3_broken given_away : ℕ) 
  (h1 : day1_total = 7)
  (h2 : day1_broken = 4)
  (h3 : day2_total = 12)
  (h4 : day2_broken = 5)
  (h5 : day3_total = 15)
  (h6 : day3_broken = 8)
  (h7 : given_away = 3) :
  day1_total - day1_broken + day2_total - day2_broken + day3_total - day3_broken - given_away = 14 := by
  sorry


end NUMINAMATH_CALUDE_toms_seashells_l4022_402213


namespace NUMINAMATH_CALUDE_intersection_forms_quadrilateral_l4022_402209

/-- A line in 2D space represented by its equation coefficients -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : (x y : ℝ) → a * x + b * y + c = 0

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a quadrilateral as a set of four distinct points -/
def IsQuadrilateral (p1 p2 p3 p4 : Point) : Prop :=
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

/-- Function to find the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- The four lines given in the problem -/
def line1 : Line := ⟨2, -1, -3, sorry⟩  -- y = 2x + 3
def line2 : Line := ⟨2, 1, -3, sorry⟩   -- y = -2x + 3
def line3 : Line := ⟨0, 1, 1, sorry⟩    -- y = -1
def line4 : Line := ⟨1, 0, -2, sorry⟩   -- x = 2

/-- Theorem stating that the intersection of the given lines forms a quadrilateral -/
theorem intersection_forms_quadrilateral :
  let p1 := intersectionPoint line1 line2
  let p2 := intersectionPoint line1 line3
  let p3 := intersectionPoint line2 line3
  let p4 := intersectionPoint line1 line4
  IsQuadrilateral p1 p2 p3 p4 := by sorry

end NUMINAMATH_CALUDE_intersection_forms_quadrilateral_l4022_402209


namespace NUMINAMATH_CALUDE_negation_equivalence_l4022_402298

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x < 0 ∧ Real.log (x^2 - 2*x - 1) ≥ 0) ↔
  (∀ x : ℝ, x < 0 → Real.log (x^2 - 2*x - 1) < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4022_402298


namespace NUMINAMATH_CALUDE_no_solution_equation_l4022_402274

theorem no_solution_equation : ¬∃ (x : ℝ), (8 / (x^2 - 4) + 1 = x / (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l4022_402274


namespace NUMINAMATH_CALUDE_fish_population_estimate_l4022_402277

/-- Estimate the number of fish in a pond using mark-recapture method -/
theorem fish_population_estimate 
  (initially_marked : ℕ) 
  (recaptured : ℕ) 
  (marked_in_recapture : ℕ) 
  (h1 : initially_marked = 2000)
  (h2 : recaptured = 500)
  (h3 : marked_in_recapture = 40) :
  (initially_marked * recaptured) / marked_in_recapture = 25000 := by
  sorry

#eval (2000 * 500) / 40

end NUMINAMATH_CALUDE_fish_population_estimate_l4022_402277


namespace NUMINAMATH_CALUDE_additional_bottles_l4022_402268

theorem additional_bottles (initial_bottles : ℕ) (capacity_per_bottle : ℕ) (total_stars : ℕ) : 
  initial_bottles = 2 → capacity_per_bottle = 15 → total_stars = 75 →
  (total_stars - initial_bottles * capacity_per_bottle) / capacity_per_bottle = 3 := by
sorry

end NUMINAMATH_CALUDE_additional_bottles_l4022_402268


namespace NUMINAMATH_CALUDE_f_increasing_on_neg_two_to_zero_l4022_402247

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem f_increasing_on_neg_two_to_zero
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_domain : Set.Icc (a - 1) 2 = Set.Icc (-2) 2) :
  StrictMonoOn (f a b) (Set.Icc (-2) 0) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_neg_two_to_zero_l4022_402247


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l4022_402269

/-- Given a hyperbola and an ellipse that share the same foci, 
    prove that the parameter m in the hyperbola equation is 7. -/
theorem hyperbola_ellipse_shared_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 8 ∧ c^2 = m + 1) → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_shared_foci_l4022_402269


namespace NUMINAMATH_CALUDE_square_equation_solution_l4022_402291

theorem square_equation_solution (a : ℝ) (h : a^2 + a^2/4 = 5) : a = 2 ∨ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l4022_402291


namespace NUMINAMATH_CALUDE_vector_not_parallel_l4022_402244

def a : ℝ × ℝ := (1, -2)

theorem vector_not_parallel (k : ℝ) : 
  ¬ ∃ (t : ℝ), (k^2 + 1, k^2 + 1) = t • a := by sorry

end NUMINAMATH_CALUDE_vector_not_parallel_l4022_402244


namespace NUMINAMATH_CALUDE_sally_quarters_l4022_402296

def quarters_problem (initial received spent : ℕ) : ℕ :=
  initial + received - spent

theorem sally_quarters : quarters_problem 760 418 152 = 1026 := by
  sorry

end NUMINAMATH_CALUDE_sally_quarters_l4022_402296


namespace NUMINAMATH_CALUDE_negation_of_proposition_ln_negation_l4022_402286

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x, p x) ↔ (∃ x, ¬ p x) :=
by sorry

theorem ln_negation :
  (¬ ∀ x : ℝ, log x > 1) ↔ (∃ x : ℝ, log x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_ln_negation_l4022_402286


namespace NUMINAMATH_CALUDE_square_value_l4022_402262

theorem square_value (x y z : ℝ) 
  (eq1 : 2*x + y + z = 17)
  (eq2 : x + 2*y + z = 14)
  (eq3 : x + y + 2*z = 13) :
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_square_value_l4022_402262
