import Mathlib

namespace NUMINAMATH_CALUDE_find_x1_l3606_360641

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + 2*(x1 - x2)^2 + 2*(x2 - x3)^2 + x3^2 = 1/2) :
  x1 = 2/3 := by sorry

end NUMINAMATH_CALUDE_find_x1_l3606_360641


namespace NUMINAMATH_CALUDE_sequence_appearance_equivalence_l3606_360690

/-- For positive real numbers a and b satisfying 2ab = a - b, 
    any positive integer n appears in the sequence (⌊ak + 1/2⌋)_{k≥1} 
    if and only if it appears at least three times in the sequence (⌊bk + 1/2⌋)_{k≥1} -/
theorem sequence_appearance_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a * b = a - b) :
  ∀ n : ℕ, n > 0 → 
    (∃ k : ℕ, k > 0 ∧ |a * k - n| < 1/2) ↔ 
    (∃ m₁ m₂ m₃ : ℕ, m₁ > 0 ∧ m₂ > 0 ∧ m₃ > 0 ∧ m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧ 
      |b * m₁ - n| < 1/2 ∧ |b * m₂ - n| < 1/2 ∧ |b * m₃ - n| < 1/2) := by
  sorry


end NUMINAMATH_CALUDE_sequence_appearance_equivalence_l3606_360690


namespace NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3606_360689

/-- A decagon is a polygon with 10 sides and 10 vertices. -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon. -/
def num_vertices : ℕ := 10

/-- The number of vertices adjacent to any given vertex in a decagon. -/
def num_adjacent_vertices : ℕ := 2

/-- The probability of selecting two adjacent vertices when choosing 2 distinct vertices at random from a decagon. -/
def prob_adjacent_vertices (d : Decagon) : ℚ :=
  num_adjacent_vertices / (num_vertices - 1)

theorem prob_adjacent_vertices_decagon :
  ∀ d : Decagon, prob_adjacent_vertices d = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_adjacent_vertices_decagon_l3606_360689


namespace NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l3606_360659

theorem cinnamon_swirls_distribution (total_pieces : ℕ) (num_people : ℕ) (pieces_per_person : ℕ) : 
  total_pieces = 12 → 
  num_people = 3 → 
  total_pieces = num_people * pieces_per_person →
  pieces_per_person = 4 := by
  sorry

end NUMINAMATH_CALUDE_cinnamon_swirls_distribution_l3606_360659


namespace NUMINAMATH_CALUDE_common_area_rectangle_circle_l3606_360645

/-- The area of the region common to a rectangle and a circle with the same center -/
theorem common_area_rectangle_circle (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_radius : ℝ) : 
  rectangle_width = 10 →
  rectangle_height = 4 →
  circle_radius = 5 →
  (rectangle_width / 2 = circle_radius) →
  (rectangle_height / 2 < circle_radius) →
  let common_area := rectangle_width * rectangle_height + 2 * π * (rectangle_height / 2)^2
  common_area = 40 + 4 * π := by
  sorry


end NUMINAMATH_CALUDE_common_area_rectangle_circle_l3606_360645


namespace NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l3606_360630

/-- Triangle PQR with sides PQ = 47, QR = 14, and RP = 50 -/
structure Triangle (P Q R : ℝ × ℝ) :=
  (pq : dist P Q = 47)
  (qr : dist Q R = 14)
  (rp : dist R P = 50)

/-- The circumcircle of triangle PQR -/
def circumcircle (P Q R : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S P = dist S Q ∧ dist S Q = dist S R}

/-- The perpendicular bisector of RP -/
def perpBisector (R P : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {S | dist S R = dist S P ∧ (S.1 - R.1) * (P.1 - R.1) + (S.2 - R.2) * (P.2 - R.2) = 0}

/-- S is on the opposite side of RP from Q -/
def oppositeSide (S Q R P : ℝ × ℝ) : Prop :=
  ((S.1 - R.1) * (P.2 - R.2) - (S.2 - R.2) * (P.1 - R.1)) *
  ((Q.1 - R.1) * (P.2 - R.2) - (Q.2 - R.2) * (P.1 - R.1)) < 0

theorem triangle_circumcircle_intersection
  (P Q R : ℝ × ℝ)
  (tri : Triangle P Q R)
  (S : ℝ × ℝ)
  (h1 : S ∈ circumcircle P Q R)
  (h2 : S ∈ perpBisector R P)
  (h3 : oppositeSide S Q R P) :
  dist P S = 8 * Real.sqrt 47 :=
sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_intersection_l3606_360630


namespace NUMINAMATH_CALUDE_karen_took_one_sixth_l3606_360660

/-- 
Given:
- Sasha added 48 cards to a box
- There were originally 43 cards in the box
- There are now 83 cards in the box

Prove that the fraction of cards Karen took out is 1/6
-/
theorem karen_took_one_sixth (cards_added : ℕ) (original_cards : ℕ) (final_cards : ℕ) 
  (h1 : cards_added = 48)
  (h2 : original_cards = 43)
  (h3 : final_cards = 83) :
  (cards_added + original_cards - final_cards : ℚ) / cards_added = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_karen_took_one_sixth_l3606_360660


namespace NUMINAMATH_CALUDE_crayon_division_l3606_360685

theorem crayon_division (total_crayons : ℕ) (num_people : ℕ) (crayons_per_person : ℕ) :
  total_crayons = 24 →
  num_people = 3 →
  crayons_per_person = total_crayons / num_people →
  crayons_per_person = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_crayon_division_l3606_360685


namespace NUMINAMATH_CALUDE_abs_x_squared_minus_4x_plus_3_lt_6_l3606_360684

theorem abs_x_squared_minus_4x_plus_3_lt_6 (x : ℝ) :
  |x^2 - 4*x + 3| < 6 ↔ 1 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_squared_minus_4x_plus_3_lt_6_l3606_360684


namespace NUMINAMATH_CALUDE_hash_eight_two_l3606_360626

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a + b)^3 * (a - b)

-- Theorem to prove
theorem hash_eight_two : hash 8 2 = 6000 := by sorry

end NUMINAMATH_CALUDE_hash_eight_two_l3606_360626


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l3606_360681

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ := a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowtie_equation_solution :
  ∀ y : ℝ, bowtie 3 y = 27 → y = 72 := by
sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l3606_360681


namespace NUMINAMATH_CALUDE_min_a_value_l3606_360658

theorem min_a_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a*x + 1 ≥ 0) →
  a ≥ -5/2 :=
sorry

end NUMINAMATH_CALUDE_min_a_value_l3606_360658


namespace NUMINAMATH_CALUDE_sector_max_area_l3606_360604

theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  (1/2) * l * r ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l3606_360604


namespace NUMINAMATH_CALUDE_parabola_directrix_l3606_360614

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix (x - 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3606_360614


namespace NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l3606_360642

theorem product_of_real_parts_of_complex_solutions : ∃ (z₁ z₂ : ℂ),
  (z₁^2 + 2*z₁ = Complex.I) ∧ 
  (z₂^2 + 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (Complex.re z₁ * Complex.re z₂ = (1 - Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_real_parts_of_complex_solutions_l3606_360642


namespace NUMINAMATH_CALUDE_school_teachers_count_l3606_360678

theorem school_teachers_count (total : ℕ) (sample_size : ℕ) (students_in_sample : ℕ) 
  (h1 : total = 2400)
  (h2 : sample_size = 150)
  (h3 : students_in_sample = 135) :
  total - (total * students_in_sample / sample_size) = 240 :=
by sorry

end NUMINAMATH_CALUDE_school_teachers_count_l3606_360678


namespace NUMINAMATH_CALUDE_solution_set_l3606_360692

theorem solution_set (x : ℝ) : 2 < x / (3 * x - 7) ∧ x / (3 * x - 7) ≤ 7 ↔ 49 / 20 < x ∧ x ≤ 14 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l3606_360692


namespace NUMINAMATH_CALUDE_eight_even_painted_cubes_l3606_360629

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Represents a cube with a certain number of painted faces -/
structure Cube where
  painted_faces : Nat

/-- Function to determine if a number is even -/
def is_even (n : Nat) : Bool :=
  n % 2 = 0

/-- Function to calculate the number of cubes with even painted faces -/
def count_even_painted_cubes (block : Block) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that a 6x3x1 block has 8 cubes with even painted faces -/
theorem eight_even_painted_cubes (block : Block) 
  (h1 : block.length = 6) 
  (h2 : block.width = 3) 
  (h3 : block.height = 1) : 
  count_even_painted_cubes block = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_even_painted_cubes_l3606_360629


namespace NUMINAMATH_CALUDE_sum_odd_integers_9_to_49_l3606_360603

/-- The sum of odd integers from 9 through 49, inclusive, is 609. -/
theorem sum_odd_integers_9_to_49 : 
  (Finset.range 21).sum (fun i => 2*i + 9) = 609 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_integers_9_to_49_l3606_360603


namespace NUMINAMATH_CALUDE_theater_casting_theorem_l3606_360647

/-- Represents the number of ways to fill roles in a theater company. -/
def theater_casting_combinations (
  female_roles : Nat
) (male_roles : Nat) (
  neutral_roles : Nat
) (auditioning_men : Nat) (
  auditioning_women : Nat
) (qualified_lead_actresses : Nat) : Nat :=
  auditioning_men *
  qualified_lead_actresses *
  (auditioning_women - qualified_lead_actresses) *
  (auditioning_women - qualified_lead_actresses - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 1) *
  (auditioning_men + auditioning_women - female_roles - male_roles - 2)

/-- Theorem stating the number of ways to fill roles in the specific theater casting scenario. -/
theorem theater_casting_theorem :
  theater_casting_combinations 3 1 3 6 7 3 = 108864 := by
  sorry

end NUMINAMATH_CALUDE_theater_casting_theorem_l3606_360647


namespace NUMINAMATH_CALUDE_farm_animals_l3606_360611

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (chickens : ℕ) (cows : ℕ) : 
  total_animals = 120 →
  total_legs = 350 →
  total_animals = chickens + cows →
  total_legs = 2 * chickens + 4 * cows →
  chickens = 65 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l3606_360611


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l3606_360639

theorem floor_ceil_sum : ⌊(1.999 : ℝ)⌋ + ⌈(3.001 : ℝ)⌉ + ⌈(0.001 : ℝ)⌉ = 6 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l3606_360639


namespace NUMINAMATH_CALUDE_sum_first_four_terms_l3606_360675

def arithmetic_sequence (a : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem sum_first_four_terms
  (a d : ℤ)
  (h5 : arithmetic_sequence a d 4 = 10)
  (h6 : arithmetic_sequence a d 5 = 14)
  (h7 : arithmetic_sequence a d 6 = 18) :
  (arithmetic_sequence a d 0) +
  (arithmetic_sequence a d 1) +
  (arithmetic_sequence a d 2) +
  (arithmetic_sequence a d 3) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_first_four_terms_l3606_360675


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_div_fifth_l3606_360623

/-- Represents a repeating decimal with a two-digit repeat -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (a * 100 + b) / 99

theorem repeating_decimal_sum_div_fifth :
  let x := RepeatingDecimal 8 3
  let y := RepeatingDecimal 1 8
  (x + y) / (1/5) = 505/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_div_fifth_l3606_360623


namespace NUMINAMATH_CALUDE_field_trip_groups_l3606_360671

/-- Given the conditions for a field trip lunch preparation, prove the number of groups. -/
theorem field_trip_groups (
  sandwiches_per_student : ℕ)
  (bread_per_sandwich : ℕ)
  (students_per_group : ℕ)
  (total_bread : ℕ)
  (h1 : sandwiches_per_student = 2)
  (h2 : bread_per_sandwich = 2)
  (h3 : students_per_group = 6)
  (h4 : total_bread = 120) :
  total_bread / (bread_per_sandwich * sandwiches_per_student * students_per_group) = 5 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_groups_l3606_360671


namespace NUMINAMATH_CALUDE_overtime_pay_fraction_l3606_360693

/-- Represents the overtime pay calculation problem --/
theorem overtime_pay_fraction (regular_wage : ℝ) (hours_per_day : ℝ) (days : ℕ) 
  (total_pay : ℝ) (regular_hours : ℝ) (overtime_fraction : ℝ) : 
  regular_wage = 18 →
  hours_per_day = 10 →
  days = 5 →
  total_pay = 990 →
  regular_hours = 8 →
  total_pay = (regular_wage * regular_hours * days) + 
    (regular_wage * (1 + overtime_fraction) * (hours_per_day - regular_hours) * days) →
  overtime_fraction = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_overtime_pay_fraction_l3606_360693


namespace NUMINAMATH_CALUDE_polynomial_symmetry_representation_l3606_360663

theorem polynomial_symmetry_representation (p : ℝ → ℝ) (a : ℝ) 
  (h_symmetry : ∀ x, p x = p (a - x)) :
  ∃ h : ℝ → ℝ, ∀ x, p x = h ((x - a / 2) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_representation_l3606_360663


namespace NUMINAMATH_CALUDE_size_relationship_l3606_360644

theorem size_relationship (a₁ a₂ b₁ b₂ : ℝ) (h1 : a₁ < a₂) (h2 : b₁ < b₂) :
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l3606_360644


namespace NUMINAMATH_CALUDE_henrys_money_l3606_360638

theorem henrys_money (x : ℤ) : 
  (x + 18 - 10 = 19) → (x = 11) := by
  sorry

end NUMINAMATH_CALUDE_henrys_money_l3606_360638


namespace NUMINAMATH_CALUDE_inequalities_on_positive_reals_l3606_360670

theorem inequalities_on_positive_reals :
  ∀ x : ℝ, x > 0 →
    (Real.log x < x) ∧
    (Real.sin x < x) ∧
    (Real.exp x > x) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_on_positive_reals_l3606_360670


namespace NUMINAMATH_CALUDE_quadratic_condition_l3606_360668

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_in_x (a b c : ℝ) : Prop := a ≠ 0

/-- The equation ax² - 2x + 3 = 0 -/
def equation (a : ℝ) (x : ℝ) : Prop := a * x^2 - 2*x + 3 = 0

theorem quadratic_condition (a : ℝ) :
  (∃ x, equation a x) ∧ is_quadratic_in_x a (-2) 3 ↔ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_condition_l3606_360668


namespace NUMINAMATH_CALUDE_second_crate_granola_weight_l3606_360653

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a crate given its dimensions -/
def crateVolume (d : CrateDimensions) : ℝ := d.height * d.width * d.length

/-- Represents the properties of the first crate -/
def firstCrate : CrateDimensions := {
  height := 4,
  width := 3,
  length := 6
}

/-- The weight of coffee the first crate can hold -/
def firstCrateWeight : ℝ := 72

/-- Represents the properties of the second crate -/
def secondCrate : CrateDimensions := {
  height := firstCrate.height * 1.5,
  width := firstCrate.width * 1.5,
  length := firstCrate.length
}

/-- Theorem stating that the second crate can hold 162 grams of granola -/
theorem second_crate_granola_weight :
  (crateVolume secondCrate / crateVolume firstCrate) * firstCrateWeight = 162 := by sorry

end NUMINAMATH_CALUDE_second_crate_granola_weight_l3606_360653


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3606_360648

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 7 = 5 / (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3606_360648


namespace NUMINAMATH_CALUDE_gcd_problem_l3606_360617

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = 2 * 2927 * k) :
  Int.gcd (3 * a^2 + 61 * a + 143) (a + 19) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3606_360617


namespace NUMINAMATH_CALUDE_function_eventually_constant_l3606_360622

def is_eventually_constant (f : ℕ+ → ℕ+) : Prop :=
  ∃ m : ℕ+, ∀ x ≥ m, f x = f m

theorem function_eventually_constant
  (f : ℕ+ → ℕ+)
  (h1 : ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1))
  (h2 : ∀ x : ℕ+, f x < 2000) :
  is_eventually_constant f := by
sorry

end NUMINAMATH_CALUDE_function_eventually_constant_l3606_360622


namespace NUMINAMATH_CALUDE_correct_change_l3606_360650

/-- The change Sandy received after buying a football and a baseball -/
def sandys_change (football_cost baseball_cost payment : ℚ) : ℚ :=
  payment - (football_cost + baseball_cost)

theorem correct_change : sandys_change 9.14 6.81 20 = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_correct_change_l3606_360650


namespace NUMINAMATH_CALUDE_cooking_time_calculation_l3606_360677

/-- Represents the cooking time for each food item -/
structure CookingTime where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Represents the quantity of each food item to be cooked -/
structure CookingQuantity where
  waffles : ℕ
  steak : ℕ
  chili : ℕ
  fries : ℕ

/-- Calculates the total cooking time given the cooking times and quantities -/
def totalCookingTime (time : CookingTime) (quantity : CookingQuantity) : ℕ :=
  time.waffles * quantity.waffles +
  time.steak * quantity.steak +
  time.chili * quantity.chili +
  time.fries * quantity.fries

/-- Theorem: Given the specified cooking times and quantities, the total cooking time is 218 minutes -/
theorem cooking_time_calculation (time : CookingTime) (quantity : CookingQuantity)
  (hw : time.waffles = 10)
  (hs : time.steak = 6)
  (hc : time.chili = 20)
  (hf : time.fries = 15)
  (qw : quantity.waffles = 5)
  (qs : quantity.steak = 8)
  (qc : quantity.chili = 3)
  (qf : quantity.fries = 4) :
  totalCookingTime time quantity = 218 := by
  sorry

end NUMINAMATH_CALUDE_cooking_time_calculation_l3606_360677


namespace NUMINAMATH_CALUDE_work_completion_original_men_l3606_360618

theorem work_completion_original_men (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 55 → absent_men = 15 → final_days = 60 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧
    original_men = 180 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_original_men_l3606_360618


namespace NUMINAMATH_CALUDE_jane_exercise_hours_per_day_l3606_360688

/-- Given Jane's exercise routine, prove the number of hours she exercises per day --/
theorem jane_exercise_hours_per_day 
  (days_per_week : ℕ) 
  (total_weeks : ℕ) 
  (total_hours : ℕ) 
  (h1 : days_per_week = 5)
  (h2 : total_weeks = 8)
  (h3 : total_hours = 40) :
  total_hours / (total_weeks * days_per_week) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jane_exercise_hours_per_day_l3606_360688


namespace NUMINAMATH_CALUDE_fifth_root_of_3125_l3606_360621

theorem fifth_root_of_3125 (x : ℝ) (h1 : x > 0) (h2 : x^5 = 3125) : x = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_3125_l3606_360621


namespace NUMINAMATH_CALUDE_seventh_term_is_seven_l3606_360687

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference of the sequence
  d : ℝ
  -- The sum of the first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- The sixth term is 6
  sixth_term : a + 5*d = 6

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_seven_l3606_360687


namespace NUMINAMATH_CALUDE_negative_sqrt_seven_greater_than_negative_sqrt_eleven_l3606_360674

theorem negative_sqrt_seven_greater_than_negative_sqrt_eleven :
  -Real.sqrt 7 > -Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_seven_greater_than_negative_sqrt_eleven_l3606_360674


namespace NUMINAMATH_CALUDE_total_marbles_is_90_l3606_360615

/-- Represents the number of marbles of each color in the bag -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of red:blue:green marbles is 2:4:6 -/
def ratio_constraint (bag : MarbleBag) : Prop :=
  3 * bag.red = bag.blue ∧ 2 * bag.blue = bag.green

/-- There are 30 blue marbles -/
def blue_constraint (bag : MarbleBag) : Prop :=
  bag.blue = 30

/-- The total number of marbles in the bag -/
def total_marbles (bag : MarbleBag) : ℕ :=
  bag.red + bag.blue + bag.green

/-- Theorem stating that the total number of marbles is 90 -/
theorem total_marbles_is_90 (bag : MarbleBag) 
  (h_ratio : ratio_constraint bag) (h_blue : blue_constraint bag) : 
  total_marbles bag = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_90_l3606_360615


namespace NUMINAMATH_CALUDE_final_sum_theorem_l3606_360609

/-- The number of participants in the game -/
def num_participants : ℕ := 42

/-- The initial value on the first calculator -/
def initial_val1 : ℤ := 2

/-- The initial value on the second calculator -/
def initial_val2 : ℤ := -2

/-- The initial value on the third calculator -/
def initial_val3 : ℤ := 3

/-- The operation performed on the first calculator -/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation performed on the second calculator -/
def op2 (n : ℤ) : ℤ := -n

/-- The operation performed on the third calculator -/
def op3 (n : ℤ) : ℤ := n ^ 3

/-- The final value on the first calculator after all iterations -/
noncomputable def final_val1 : ℤ := initial_val1 ^ (2 ^ num_participants)

/-- The final value on the second calculator after all iterations -/
def final_val2 : ℤ := initial_val2

/-- The final value on the third calculator after all iterations -/
noncomputable def final_val3 : ℤ := initial_val3 ^ (3 ^ num_participants)

/-- The theorem stating the sum of the final values on all calculators -/
theorem final_sum_theorem : 
  final_val1 + final_val2 + final_val3 = 2^(2^num_participants) - 2 + 3^(3^num_participants) :=
by sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l3606_360609


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3606_360665

theorem quadratic_transformation (a b : ℝ) :
  (∀ x : ℝ, x^2 - 10*x + b = (x - a)^2 - 1) → b - a = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3606_360665


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_l3606_360607

theorem circle_equation_from_diameter (P Q : ℝ × ℝ) : 
  P = (4, 0) → Q = (0, 2) → 
  ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 5 ↔ 
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
      x = 4 * (1 - t) + 0 * t ∧ 
      y = 0 * (1 - t) + 2 * t ∧
      (x - 4)^2 + (y - 0)^2 = (0 - 4)^2 + (2 - 0)^2 / 4) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_l3606_360607


namespace NUMINAMATH_CALUDE_linear_function_property_l3606_360666

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- Given a linear function g where g(5) - g(1) = 16, prove that g(13) - g(1) = 48. -/
theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) 
  (h_given : g 5 - g 1 = 16) : 
  g 13 - g 1 = 48 := by
  sorry


end NUMINAMATH_CALUDE_linear_function_property_l3606_360666


namespace NUMINAMATH_CALUDE_cds_per_rack_l3606_360624

/-- Given a shelf that can hold 4 racks and 32 CDs, prove that each rack can hold 8 CDs. -/
theorem cds_per_rack (racks_per_shelf : ℕ) (cds_per_shelf : ℕ) (h1 : racks_per_shelf = 4) (h2 : cds_per_shelf = 32) :
  cds_per_shelf / racks_per_shelf = 8 := by
  sorry


end NUMINAMATH_CALUDE_cds_per_rack_l3606_360624


namespace NUMINAMATH_CALUDE_no_article_before_word_l3606_360654

-- Define the sentence structure
def sentence_structure : String := "They sent us ______ word of the latest happenings."

-- Define the function to determine the correct article
def correct_article : String := ""

-- Theorem statement
theorem no_article_before_word :
  correct_article = "" := by sorry

end NUMINAMATH_CALUDE_no_article_before_word_l3606_360654


namespace NUMINAMATH_CALUDE_inequality_proof_l3606_360633

theorem inequality_proof (a b c A B C k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_A : 0 < A) (pos_B : 0 < B) (pos_C : 0 < C)
  (sum_a : a + A = k) (sum_b : b + B = k) (sum_c : c + C = k) : 
  a * B + b * C + c * A < k^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3606_360633


namespace NUMINAMATH_CALUDE_students_passed_l3606_360695

theorem students_passed (total : ℕ) (fail_freq : ℚ) (h1 : total = 1000) (h2 : fail_freq = 0.4) :
  total - (total * fail_freq).floor = 600 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l3606_360695


namespace NUMINAMATH_CALUDE_diamond_royalty_sanity_undetermined_l3606_360602

/-- Represents the sanity status of a person -/
inductive SanityStatus
  | Sane
  | Insane
  | Unknown

/-- Represents a royal person -/
structure RoyalPerson where
  name : String
  status : SanityStatus

/-- Represents a rumor about a person's sanity -/
structure Rumor where
  subject : RoyalPerson
  content : SanityStatus

/-- Represents the reliability of information -/
inductive Reliability
  | Reliable
  | Unreliable
  | Unknown

/-- The problem setup -/
def diamondRoyalty : Prop := ∃ (king queen : RoyalPerson) 
  (rumor : Rumor) (rumorReliability : Reliability),
  king.name = "King of Diamonds" ∧
  queen.name = "Queen of Diamonds" ∧
  rumor.subject = queen ∧
  rumor.content = SanityStatus.Insane ∧
  rumorReliability = Reliability.Unknown ∧
  (king.status = SanityStatus.Unknown ∨ 
   king.status = SanityStatus.Insane) ∧
  queen.status = SanityStatus.Unknown

/-- The theorem to be proved -/
theorem diamond_royalty_sanity_undetermined : 
  diamondRoyalty → 
  ∃ (king queen : RoyalPerson), 
    king.name = "King of Diamonds" ∧
    queen.name = "Queen of Diamonds" ∧
    king.status = SanityStatus.Unknown ∧
    queen.status = SanityStatus.Unknown :=
by
  sorry

end NUMINAMATH_CALUDE_diamond_royalty_sanity_undetermined_l3606_360602


namespace NUMINAMATH_CALUDE_nes_sale_price_l3606_360625

theorem nes_sale_price 
  (snes_value : ℝ)
  (trade_in_percentage : ℝ)
  (additional_cash : ℝ)
  (change : ℝ)
  (game_value : ℝ)
  (h1 : snes_value = 150)
  (h2 : trade_in_percentage = 0.8)
  (h3 : additional_cash = 80)
  (h4 : change = 10)
  (h5 : game_value = 30) :
  snes_value * trade_in_percentage + additional_cash - change - game_value = 160 :=
by
  sorry

#check nes_sale_price

end NUMINAMATH_CALUDE_nes_sale_price_l3606_360625


namespace NUMINAMATH_CALUDE_tim_one_dollar_bills_l3606_360627

/-- Represents the number of bills of a certain denomination -/
structure BillCount where
  count : ℕ
  denomination : ℕ

/-- Represents Tim's wallet -/
structure Wallet where
  tenDollarBills : BillCount
  fiveDollarBills : BillCount
  oneDollarBills : BillCount

def Wallet.totalValue (w : Wallet) : ℕ :=
  w.tenDollarBills.count * w.tenDollarBills.denomination +
  w.fiveDollarBills.count * w.fiveDollarBills.denomination +
  w.oneDollarBills.count * w.oneDollarBills.denomination

def Wallet.totalBills (w : Wallet) : ℕ :=
  w.tenDollarBills.count + w.fiveDollarBills.count + w.oneDollarBills.count

theorem tim_one_dollar_bills 
  (w : Wallet)
  (h1 : w.tenDollarBills = ⟨13, 10⟩)
  (h2 : w.fiveDollarBills = ⟨11, 5⟩)
  (h3 : w.totalValue = 128)
  (h4 : w.totalBills ≥ 16) :
  w.oneDollarBills.count = 57 := by
  sorry

end NUMINAMATH_CALUDE_tim_one_dollar_bills_l3606_360627


namespace NUMINAMATH_CALUDE_power_mod_nine_l3606_360673

theorem power_mod_nine (x : ℤ) : x = 5 → x^46655 % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_nine_l3606_360673


namespace NUMINAMATH_CALUDE_floor_painted_by_all_colors_l3606_360664

/-- Represents the percentage of floor painted by each painter -/
structure PainterCoverage where
  red : Real
  green : Real
  blue : Real

/-- Theorem: Given the paint coverage, at least 10% of the floor is painted by all three colors -/
theorem floor_painted_by_all_colors (coverage : PainterCoverage) 
  (h_red : coverage.red = 75)
  (h_green : coverage.green = -70)
  (h_blue : coverage.blue = -65) :
  ∃ (all_colors_coverage : Real),
    all_colors_coverage ≥ 10 ∧ 
    all_colors_coverage ≤ 100 ∧
    all_colors_coverage ≤ coverage.red ∧
    all_colors_coverage ≤ -coverage.green ∧
    all_colors_coverage ≤ -coverage.blue :=
sorry

end NUMINAMATH_CALUDE_floor_painted_by_all_colors_l3606_360664


namespace NUMINAMATH_CALUDE_employment_percentage_l3606_360637

theorem employment_percentage (population : ℝ) (employed : ℝ) 
  (h1 : employed > 0) 
  (h2 : population > 0) 
  (h3 : 0.42 * population = 0.7 * employed) : 
  employed / population = 0.6 := by
sorry

end NUMINAMATH_CALUDE_employment_percentage_l3606_360637


namespace NUMINAMATH_CALUDE_cement_price_per_bag_l3606_360661

theorem cement_price_per_bag 
  (cement_bags : ℕ) 
  (sand_lorries : ℕ) 
  (sand_tons_per_lorry : ℕ) 
  (sand_price_per_ton : ℕ) 
  (total_payment : ℕ) 
  (h1 : cement_bags = 500)
  (h2 : sand_lorries = 20)
  (h3 : sand_tons_per_lorry = 10)
  (h4 : sand_price_per_ton = 40)
  (h5 : total_payment = 13000) :
  (total_payment - sand_lorries * sand_tons_per_lorry * sand_price_per_ton) / cement_bags = 10 :=
by sorry

end NUMINAMATH_CALUDE_cement_price_per_bag_l3606_360661


namespace NUMINAMATH_CALUDE_oil_floats_on_water_l3606_360640

-- Define the density of a substance
def density (substance : Type) : ℝ := sorry

-- Define what it means for a substance to float on another
def floats_on (a b : Type) : Prop := 
  density a < density b

-- Define oil and water as types
def oil : Type := sorry
def water : Type := sorry

-- State the theorem
theorem oil_floats_on_water : 
  (density oil < density water) → floats_on oil water := by sorry

end NUMINAMATH_CALUDE_oil_floats_on_water_l3606_360640


namespace NUMINAMATH_CALUDE_remainder_theorem_l3606_360679

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^5 - 2*x^4 - x^3 + 2*x^2 + x

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := (x^2 - 9) * (x - 1)

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 9*x^2 + 73*x - 81

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3606_360679


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3606_360632

def M : Set ℝ := {x | 0 < x ∧ x < 1}
def N : Set ℝ := {x | -2 < x ∧ x < 1}

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3606_360632


namespace NUMINAMATH_CALUDE_power_inequality_l3606_360652

theorem power_inequality : 0.1^0.8 < 0.2^0.8 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3606_360652


namespace NUMINAMATH_CALUDE_circumscribed_sphere_volume_l3606_360696

theorem circumscribed_sphere_volume (cube_surface_area : ℝ) (h : cube_surface_area = 24) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := cube_edge * Real.sqrt 3 / 2
  (4 / 3) * Real.pi * sphere_radius ^ 3 = 4 * Real.sqrt 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_volume_l3606_360696


namespace NUMINAMATH_CALUDE_triangle_side_length_l3606_360694

theorem triangle_side_length (a c area : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_a : a = 4) (h_c : c = 6) (h_area : area = 6 * Real.sqrt 3) : 
    ∃ (b : ℝ), b^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3606_360694


namespace NUMINAMATH_CALUDE_problem1_problem1_evaluation_l3606_360610

theorem problem1 (x : ℝ) : 
  3 * x^3 - (x^3 + (6 * x^2 - 7 * x)) - 2 * (x^3 - 3 * x^2 - 4 * x) = 15 * x :=
by sorry

theorem problem1_evaluation : 
  3 * (-1)^3 - ((-1)^3 + (6 * (-1)^2 - 7 * (-1))) - 2 * ((-1)^3 - 3 * (-1)^2 - 4 * (-1)) = -15 :=
by sorry

end NUMINAMATH_CALUDE_problem1_problem1_evaluation_l3606_360610


namespace NUMINAMATH_CALUDE_bells_lcm_l3606_360646

def church_interval : ℕ := 18
def school_interval : ℕ := 24
def city_hall_interval : ℕ := 30

theorem bells_lcm :
  Nat.lcm (Nat.lcm church_interval school_interval) city_hall_interval = 360 := by
  sorry

end NUMINAMATH_CALUDE_bells_lcm_l3606_360646


namespace NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l3606_360649

/-- Represents the points on the circle --/
inductive Point
  | one
  | two
  | three
  | four
  | five
  | six

/-- Determines if a point is odd-numbered --/
def isOdd (p : Point) : Bool :=
  match p with
  | Point.one | Point.three | Point.five => true
  | _ => false

/-- Calculates the next point after a jump --/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.three
  | Point.two => Point.three
  | Point.three => Point.five
  | Point.four => Point.five
  | Point.five => Point.one
  | Point.six => Point.one

/-- Calculates the point after n jumps --/
def jumpN (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpN start n)

theorem bug_position_after_2023_jumps :
  jumpN Point.six 2023 = Point.one := by
  sorry

end NUMINAMATH_CALUDE_bug_position_after_2023_jumps_l3606_360649


namespace NUMINAMATH_CALUDE_fibonacci_sequence_contains_one_l3606_360616

-- Define Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the sequence x_n
def x (k m : ℕ) : ℕ → ℚ
  | 0 => (fib k : ℚ) / (fib m : ℚ)
  | (n + 1) =>
      let xn := x k m n
      if xn = 1 then 1 else (2 * xn - 1) / (1 - xn)

-- Main theorem
theorem fibonacci_sequence_contains_one (k m : ℕ) (hk : k > 0) (hm : m > k) :
  (∃ n, x k m n = 1) ↔ ∃ t : ℕ, k = 2 * t + 1 ∧ m = 2 * t + 2 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_sequence_contains_one_l3606_360616


namespace NUMINAMATH_CALUDE_inequality_preservation_l3606_360643

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3606_360643


namespace NUMINAMATH_CALUDE_factor_x12_minus_1024_l3606_360672

theorem factor_x12_minus_1024 (x : ℝ) : 
  x^12 - 1024 = (x^6 + 32) * (x^3 + 4 * Real.sqrt 2) * (x^3 - 4 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_x12_minus_1024_l3606_360672


namespace NUMINAMATH_CALUDE_annika_hike_distance_l3606_360699

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  flat_speed : ℝ  -- minutes per kilometer on flat terrain
  uphill_speed : ℝ  -- minutes per kilometer uphill
  downhill_speed : ℝ  -- minutes per kilometer downhill
  initial_distance : ℝ  -- kilometers hiked initially
  total_time : ℝ  -- total time available to return

/-- Calculates the total distance hiked east given a hiking scenario -/
def total_distance_east (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating the total distance hiked east in the given scenario -/
theorem annika_hike_distance (scenario : HikingScenario) 
  (h1 : scenario.flat_speed = 10)
  (h2 : scenario.uphill_speed = 15)
  (h3 : scenario.downhill_speed = 7)
  (h4 : scenario.initial_distance = 2.5)
  (h5 : scenario.total_time = 35) :
  total_distance_east scenario = 3.0833 :=
sorry

end NUMINAMATH_CALUDE_annika_hike_distance_l3606_360699


namespace NUMINAMATH_CALUDE_monomial_2023_matches_pattern_l3606_360656

/-- Represents a monomial in the sequence -/
def monomial (n : ℕ) : ℚ × ℕ := ((2 * n + 1) / n, n)

/-- The 2023rd monomial in the sequence -/
def monomial_2023 : ℚ × ℕ := (4047 / 2023, 2023)

/-- Theorem stating that the 2023rd monomial matches the pattern -/
theorem monomial_2023_matches_pattern : monomial 2023 = monomial_2023 := by
  sorry

end NUMINAMATH_CALUDE_monomial_2023_matches_pattern_l3606_360656


namespace NUMINAMATH_CALUDE_decimal_division_l3606_360686

theorem decimal_division (x y : ℚ) (hx : x = 45/100) (hy : y = 5/1000) : x / y = 90 := by
  sorry

end NUMINAMATH_CALUDE_decimal_division_l3606_360686


namespace NUMINAMATH_CALUDE_sqrt_of_four_l3606_360698

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l3606_360698


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3606_360631

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 8 →
  c = 8 * Real.sqrt 3 →
  S = 16 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  A = π/6 ∨ A = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3606_360631


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3606_360613

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + m = 0 ∧ s^2 - 4*s + m = 0) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l3606_360613


namespace NUMINAMATH_CALUDE_complex_number_problem_l3606_360676

theorem complex_number_problem (a : ℝ) :
  (((a^2 - 1) : ℂ) + (a + 1) * I).im ≠ 0 ∧ ((a^2 - 1) : ℂ).re = 0 →
  (a + I^2016) / (1 + I) = 1 - I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3606_360676


namespace NUMINAMATH_CALUDE_man_son_age_difference_l3606_360601

/-- Represents the age difference between a man and his son -/
def ageDifference (manAge sonAge : ℕ) : ℕ := manAge - sonAge

theorem man_son_age_difference :
  ∀ (manAge sonAge : ℕ),
  sonAge = 14 →
  manAge + 2 = 2 * (sonAge + 2) →
  ageDifference manAge sonAge = 16 := by
sorry

end NUMINAMATH_CALUDE_man_son_age_difference_l3606_360601


namespace NUMINAMATH_CALUDE_connie_grandmother_birth_year_l3606_360606

/-- Calculates the birth year of Connie's grandmother given the birth years of her siblings and the gap condition. -/
def grandmotherBirthYear (brotherBirthYear sisterBirthYear : ℕ) : ℕ :=
  let siblingGap := sisterBirthYear - brotherBirthYear
  sisterBirthYear - 2 * siblingGap

/-- Proves that Connie's grandmother was born in 1928 given the known conditions. -/
theorem connie_grandmother_birth_year :
  grandmotherBirthYear 1932 1936 = 1928 := by
  sorry

#eval grandmotherBirthYear 1932 1936

end NUMINAMATH_CALUDE_connie_grandmother_birth_year_l3606_360606


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3606_360605

/-- Given non-zero numbers a, b, c such that ax^2 + bx + c > cx for all real x,
    prove that cx^2 - bx + a > cx - b for all real x. -/
theorem quadratic_inequality (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_given : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3606_360605


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l3606_360620

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l3606_360620


namespace NUMINAMATH_CALUDE_smallest_number_proof_l3606_360628

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_single : a < 10
  b_single : b < 10
  c_single : c < 10
  d_single : d < 10

/-- Checks if the given four-digit number satisfies the product conditions -/
def satisfies_conditions (n : FourDigitNumber) : Prop :=
  (n.a * n.b = 21 ∧ n.b * n.c = 20) ∨
  (n.a * n.b = 21 ∧ n.c * n.d = 20) ∨
  (n.b * n.c = 21 ∧ n.c * n.d = 20)

/-- The smallest four-digit number satisfying the conditions -/
def smallest_satisfying_number : FourDigitNumber :=
  { a := 3, b := 7, c := 4, d := 5,
    a_single := by norm_num,
    b_single := by norm_num,
    c_single := by norm_num,
    d_single := by norm_num }

theorem smallest_number_proof :
  satisfies_conditions smallest_satisfying_number ∧
  ∀ n : FourDigitNumber, satisfies_conditions n →
    n.a * 1000 + n.b * 100 + n.c * 10 + n.d ≥
    smallest_satisfying_number.a * 1000 +
    smallest_satisfying_number.b * 100 +
    smallest_satisfying_number.c * 10 +
    smallest_satisfying_number.d :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l3606_360628


namespace NUMINAMATH_CALUDE_pierre_cake_consumption_l3606_360680

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (nathalie_parts : ℝ) (pierre_multiplier : ℝ) : 
  cake_weight = 400 → 
  num_parts = 8 → 
  nathalie_parts = 1 / 8 → 
  pierre_multiplier = 2 → 
  pierre_multiplier * (nathalie_parts * cake_weight) = 100 := by
  sorry

end NUMINAMATH_CALUDE_pierre_cake_consumption_l3606_360680


namespace NUMINAMATH_CALUDE_fraction_sum_equals_negative_two_l3606_360634

theorem fraction_sum_equals_negative_two (a b : ℝ) (h1 : a + b = 0) (h2 : a * b ≠ 0) :
  b / a + a / b = -2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_negative_two_l3606_360634


namespace NUMINAMATH_CALUDE_gcd_n4_plus_16_n_plus_3_l3606_360655

theorem gcd_n4_plus_16_n_plus_3 (n : ℕ) (h : n > 16) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n4_plus_16_n_plus_3_l3606_360655


namespace NUMINAMATH_CALUDE_yard_sale_books_bought_l3606_360600

/-- The number of books Mike bought at a yard sale -/
def books_bought (initial_books final_books : ℕ) : ℕ :=
  final_books - initial_books

/-- Theorem: The number of books Mike bought at the yard sale is the difference between his final and initial number of books -/
theorem yard_sale_books_bought (initial_books final_books : ℕ) 
  (h : final_books ≥ initial_books) : 
  books_bought initial_books final_books = final_books - initial_books :=
by
  sorry

/-- Given Mike's initial and final number of books, calculate how many he bought -/
def mikes_books : ℕ := 
  books_bought 35 56

#eval mikes_books

end NUMINAMATH_CALUDE_yard_sale_books_bought_l3606_360600


namespace NUMINAMATH_CALUDE_problem_solution_l3606_360697

-- Define the propositions
def proposition_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (2*a - 1)*x + a^2 > 0

def proposition_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (a^2 - 1)^x > (a^2 - 1)^y

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, (proposition_A a ∨ proposition_B a) ↔ (a < -1 ∧ a > -Real.sqrt 2) ∨ a > 1/4) ∧
  (∀ a : ℝ, a < -1 ∧ a > -Real.sqrt 2 → a^3 + 1 < a^2 + a) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3606_360697


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_property_l3606_360612

/-- Given an ellipse and a hyperbola with shared foci, prove a property of the ellipse's semi-minor axis --/
theorem ellipse_hyperbola_property (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
   ∃ x' y' : ℝ, x'^2 - y'^2/4 = 1 ∧ 
   ∃ A B : ℝ × ℝ, 
     (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*a)^2 ∧
     (∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
       (t*A.1 + (1-t)*B.1)^2/a^2 + (t*A.2 + (1-t)*B.2)^2/b^2 = 1 ∧
       ((1-t)*A.1 + t*B.1)^2/a^2 + ((1-t)*A.2 + t*B.2)^2/b^2 = 1 ∧
       t = 1/3)) →
  b^2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_property_l3606_360612


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3606_360667

theorem smallest_positive_integer_with_remainders : 
  ∃ n : ℕ, n > 1 ∧ 
    n % 5 = 1 ∧ 
    n % 7 = 1 ∧ 
    n % 8 = 1 ∧ 
    (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) ∧
    80 < n ∧ 
    n < 299 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l3606_360667


namespace NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3606_360662

/-- Given an ellipse and a parabola with a common point, this theorem proves the range of parameter a. -/
theorem ellipse_parabola_intersection_range :
  ∀ (a x y : ℝ),
  (x^2 + 4*(y - a)^2 = 4) →  -- Ellipse equation
  (x^2 = 2*y) →              -- Parabola equation
  (-1 ≤ a ∧ a ≤ 17/8) :=     -- Range of a
by sorry

end NUMINAMATH_CALUDE_ellipse_parabola_intersection_range_l3606_360662


namespace NUMINAMATH_CALUDE_unique_geometric_sequence_value_l3606_360635

/-- Two geometric sequences with specific conditions -/
structure GeometricSequences (a : ℝ) :=
  (a_seq : ℕ → ℝ)
  (b_seq : ℕ → ℝ)
  (a_positive : a > 0)
  (a_first : a_seq 1 = a)
  (b_minus_a_1 : b_seq 1 - a_seq 1 = 1)
  (b_minus_a_2 : b_seq 2 - a_seq 2 = 2)
  (b_minus_a_3 : b_seq 3 - a_seq 3 = 3)
  (a_geometric : ∀ n : ℕ, a_seq (n + 1) / a_seq n = a_seq 2 / a_seq 1)
  (b_geometric : ∀ n : ℕ, b_seq (n + 1) / b_seq n = b_seq 2 / b_seq 1)

/-- If the a_seq is unique, then a = 1/3 -/
theorem unique_geometric_sequence_value (a : ℝ) (h : GeometricSequences a) 
  (h_unique : ∃! q : ℝ, ∀ n : ℕ, h.a_seq (n + 1) = h.a_seq n * q) : 
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_unique_geometric_sequence_value_l3606_360635


namespace NUMINAMATH_CALUDE_specific_ap_first_term_l3606_360683

/-- An arithmetic progression with given parameters -/
structure ArithmeticProgression where
  n : ℕ             -- number of terms
  d : ℤ             -- common difference
  last_term : ℤ     -- last term

/-- The first term of an arithmetic progression -/
def first_term (ap : ArithmeticProgression) : ℤ :=
  ap.last_term - (ap.n - 1) * ap.d

/-- Theorem stating the first term of the specific arithmetic progression -/
theorem specific_ap_first_term :
  let ap : ArithmeticProgression := ⟨31, 2, 62⟩
  first_term ap = 2 := by sorry

end NUMINAMATH_CALUDE_specific_ap_first_term_l3606_360683


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3606_360608

/-- Given a point P with coordinates (3,2), prove that its symmetrical point
    with respect to the x-axis has coordinates (3,-2) -/
theorem symmetry_wrt_x_axis :
  let P : ℝ × ℝ := (3, 2)
  let symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetry_x P = (3, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l3606_360608


namespace NUMINAMATH_CALUDE_milk_problem_l3606_360669

theorem milk_problem (M : ℝ) : 
  M > 0 → 
  (1 - 2/3) * (1 - 2/5) * (1 - 1/6) * M = 120 → 
  M = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l3606_360669


namespace NUMINAMATH_CALUDE_min_value_arithmetic_seq_l3606_360691

/-- An arithmetic sequence with positive terms and a_4 = 5 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 4 = 5

/-- The minimum value of 1/a_2 + 16/a_6 for the given arithmetic sequence -/
theorem min_value_arithmetic_seq (a : ℕ → ℝ) (h : ArithmeticSequence a) :
    (∀ n, a n > 0) → (1 / a 2 + 16 / a 6) ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_arithmetic_seq_l3606_360691


namespace NUMINAMATH_CALUDE_freds_salary_l3606_360682

theorem freds_salary (mikes_current_salary : ℝ) (mikes_salary_ratio : ℝ) (salary_increase_percent : ℝ) :
  mikes_current_salary = 15400 ∧
  mikes_salary_ratio = 10 ∧
  salary_increase_percent = 40 →
  (mikes_current_salary / (1 + salary_increase_percent / 100) / mikes_salary_ratio) = 1100 :=
by sorry

end NUMINAMATH_CALUDE_freds_salary_l3606_360682


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l3606_360636

theorem intersection_implies_sum (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  A ∩ B = {2} → a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l3606_360636


namespace NUMINAMATH_CALUDE_problem_solution_l3606_360619

theorem problem_solution (x : ℝ) 
  (h : (4:ℝ)^(2*x) + (2:ℝ)^(-x) + 1 = (129 + 8*Real.sqrt 2) * ((4:ℝ)^x + (2:ℝ)^(-x) - (2:ℝ)^x)) :
  10 * x = 35 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3606_360619


namespace NUMINAMATH_CALUDE_problem_solution_l3606_360651

theorem problem_solution (x n : ℝ) (h1 : x = 40) (h2 : ((x / 4) * 5) + n - 12 = 48) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3606_360651


namespace NUMINAMATH_CALUDE_conditional_probability_l3606_360657

theorem conditional_probability (P_AB P_A : ℝ) (h1 : P_AB = 2/15) (h2 : P_A = 2/5) :
  P_AB / P_A = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_l3606_360657
