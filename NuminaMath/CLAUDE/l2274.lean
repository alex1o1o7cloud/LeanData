import Mathlib

namespace NUMINAMATH_CALUDE_canteen_to_bathroom_ratio_l2274_227445

/-- Represents the number of tables in the classroom -/
def num_tables : ℕ := 6

/-- Represents the number of students currently sitting at each table -/
def students_per_table : ℕ := 3

/-- Represents the number of girls who went to the bathroom -/
def girls_in_bathroom : ℕ := 3

/-- Represents the number of new groups added to the class -/
def new_groups : ℕ := 2

/-- Represents the number of students in each new group -/
def students_per_new_group : ℕ := 4

/-- Represents the number of countries from which foreign exchange students came -/
def num_countries : ℕ := 3

/-- Represents the number of foreign exchange students from each country -/
def students_per_country : ℕ := 3

/-- Represents the total number of students supposed to be in the class -/
def total_students : ℕ := 47

/-- Theorem stating the ratio of students who went to the canteen to girls who went to the bathroom -/
theorem canteen_to_bathroom_ratio :
  let students_present := num_tables * students_per_table
  let new_group_students := new_groups * students_per_new_group
  let foreign_students := num_countries * students_per_country
  let missing_students := girls_in_bathroom + new_group_students + foreign_students
  let canteen_students := total_students - students_present - missing_students
  (canteen_students : ℚ) / girls_in_bathroom = 3 := by
  sorry

end NUMINAMATH_CALUDE_canteen_to_bathroom_ratio_l2274_227445


namespace NUMINAMATH_CALUDE_adults_group_size_l2274_227485

/-- The number of children in each group -/
def children_per_group : ℕ := 15

/-- The minimum number of adults (and children) attending -/
def min_attendees : ℕ := 255

/-- The number of adults in each group -/
def adults_per_group : ℕ := 15

theorem adults_group_size :
  (min_attendees % children_per_group = 0) →
  (min_attendees % adults_per_group = 0) →
  (min_attendees / children_per_group = min_attendees / adults_per_group) →
  adults_per_group = 15 := by
  sorry

end NUMINAMATH_CALUDE_adults_group_size_l2274_227485


namespace NUMINAMATH_CALUDE_last_locker_opened_l2274_227428

/-- Represents the locker opening pattern described in the problem -/
def lockerOpeningPattern (n : ℕ) : ℕ → Prop :=
  sorry

/-- The number of lockers -/
def totalLockers : ℕ := 2048

/-- Theorem stating that the last locker opened is number 2046 -/
theorem last_locker_opened :
  ∃ (last : ℕ), last = 2046 ∧ 
  (∀ (k : ℕ), k ≤ totalLockers → lockerOpeningPattern totalLockers k → k ≤ last) ∧
  lockerOpeningPattern totalLockers last :=
sorry

end NUMINAMATH_CALUDE_last_locker_opened_l2274_227428


namespace NUMINAMATH_CALUDE_problem_solution_l2274_227441

theorem problem_solution (x : ℝ) : x = 22.142857142857142 →
  2 * ((((x + 5) * 7) / 5) - 5) = 66 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2274_227441


namespace NUMINAMATH_CALUDE_square_root_problem_l2274_227461

theorem square_root_problem (x : ℝ) : 
  Real.sqrt x = 3.87 → x = 14.9769 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l2274_227461


namespace NUMINAMATH_CALUDE_fraction_equality_implies_value_l2274_227453

theorem fraction_equality_implies_value (b : ℝ) :
  b / (b + 30) = 0.92 → b = 345 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_value_l2274_227453


namespace NUMINAMATH_CALUDE_pennies_in_change_l2274_227487

def bread_price : ℚ := 4.79
def cheese_price : ℚ := 6.55
def milk_price : ℚ := 3.85
def strawberries_price : ℚ := 2.15
def bread_quantity : ℕ := 3
def cheese_quantity : ℕ := 2
def milk_quantity : ℕ := 6
def strawberries_quantity : ℕ := 4
def amount_given : ℚ := 100.00
def tax_rate : ℚ := 0.065
def quarters_available : ℕ := 5
def dimes_available : ℕ := 10
def nickels_available : ℕ := 15

def total_cost : ℚ :=
  (bread_price * bread_quantity +
   cheese_price * cheese_quantity +
   milk_price * milk_quantity +
   strawberries_price * strawberries_quantity) *
  (1 + tax_rate)

def change : ℚ := amount_given - total_cost

def quarters_value : ℚ := 0.25 * quarters_available
def dimes_value : ℚ := 0.10 * dimes_available
def nickels_value : ℚ := 0.05 * nickels_available

def remaining_change : ℚ := change - quarters_value - dimes_value - nickels_value

theorem pennies_in_change : 
  (remaining_change * 100).floor = 3398 := by sorry

end NUMINAMATH_CALUDE_pennies_in_change_l2274_227487


namespace NUMINAMATH_CALUDE_power_of_two_equation_l2274_227452

theorem power_of_two_equation (x : ℕ) : 16^3 + 16^3 + 16^3 = 2^x ↔ x = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l2274_227452


namespace NUMINAMATH_CALUDE_alex_age_theorem_l2274_227482

theorem alex_age_theorem :
  ∃! x : ℕ, x > 0 ∧ x ≤ 100 ∧ 
  ∃ y : ℕ, x - 2 = y^2 ∧
  ∃ z : ℕ, x + 2 = z^3 :=
by
  sorry

end NUMINAMATH_CALUDE_alex_age_theorem_l2274_227482


namespace NUMINAMATH_CALUDE_meaningful_range_l2274_227433

def is_meaningful (x : ℝ) : Prop :=
  x - 1 ≥ 0 ∧ x ≠ 3

theorem meaningful_range : 
  ∀ x : ℝ, is_meaningful x ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_l2274_227433


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2274_227457

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 6 = 12 →
  a 2 + a 9 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2274_227457


namespace NUMINAMATH_CALUDE_intersection_point_l2274_227472

-- Define the linear functions
def f1 (a b x : ℝ) : ℝ := a * x + b + 3
def f2 (a b x : ℝ) : ℝ := -b * x + a - 2
def f3 (x : ℝ) : ℝ := 2 * x - 8

-- State the theorem
theorem intersection_point (a b : ℝ) :
  (∃ y, f1 a b 0 = f2 a b 0 ∧ y = f1 a b 0) ∧  -- First and second functions intersect on y-axis
  (∃ x, f2 a b x = f3 x ∧ f2 a b x = 0) →      -- Second and third functions intersect on x-axis
  (∃ x y, f1 a b x = f3 x ∧ y = f1 a b x ∧ x = -3 ∧ y = -14) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2274_227472


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2274_227473

theorem fraction_evaluation (a b : ℝ) (h1 : a = 5) (h2 : b = 3) :
  3 / (a + b) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2274_227473


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2274_227486

theorem negation_of_proposition (n : ℕ) :
  ¬(2^n > 1000) ↔ (2^n ≤ 1000) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2274_227486


namespace NUMINAMATH_CALUDE_find_vector_c_l2274_227497

/-- Given vectors a and b in ℝ², find vector c satisfying the given conditions -/
theorem find_vector_c (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 2)) : 
  ∃ c : ℝ × ℝ, 
    (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) ∧ 
    (∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2)) → 
    c = (7/3, 7/9) := by
  sorry

end NUMINAMATH_CALUDE_find_vector_c_l2274_227497


namespace NUMINAMATH_CALUDE_min_value_ab_minus_cd_l2274_227408

theorem min_value_ab_minus_cd (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 9)
  (h5 : a^2 + b^2 + c^2 + d^2 = 21) :
  a * b - c * d ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_minus_cd_l2274_227408


namespace NUMINAMATH_CALUDE_gcd_of_f_is_2730_l2274_227499

-- Define the function f(n) = n^13 - n
def f (n : ℤ) : ℤ := n^13 - n

-- State the theorem
theorem gcd_of_f_is_2730 : 
  ∃ (d : ℕ), d = 2730 ∧ ∀ (n : ℤ), (f n).natAbs ∣ d ∧ 
  (∀ (m : ℕ), (∀ (k : ℤ), (f k).natAbs ∣ m) → d ∣ m) :=
sorry

end NUMINAMATH_CALUDE_gcd_of_f_is_2730_l2274_227499


namespace NUMINAMATH_CALUDE_sum_a_b_is_one_third_l2274_227419

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The theorem stating that a + b = 1/3 given the conditions -/
theorem sum_a_b_is_one_third
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + 3 * a + b)
  (h2 : IsEven f)
  (h3 : Set.Icc (a - 1) (2 * a) = Set.range f) :
  a + b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_is_one_third_l2274_227419


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_40_l2274_227438

/-- The maximum area of a rectangle with a perimeter of 40 units is 100 square units. -/
theorem max_area_rectangle_with_perimeter_40 :
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = 40 ∧
    length * width = 100 ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → 2 * (l + w) = 40 → l * w ≤ 100 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_40_l2274_227438


namespace NUMINAMATH_CALUDE_dino_money_theorem_l2274_227443

/-- Calculates Dino's remaining money at the end of the month -/
def dino_remaining_money (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℕ) (expenses : ℕ) : ℕ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 - expenses

/-- Theorem: Dino's remaining money at the end of the month is $500 -/
theorem dino_money_theorem : dino_remaining_money 20 30 5 10 20 40 500 = 500 := by
  sorry

end NUMINAMATH_CALUDE_dino_money_theorem_l2274_227443


namespace NUMINAMATH_CALUDE_parents_age_difference_l2274_227431

/-- The difference between Sobha's parents' ages -/
def age_difference (s : ℕ) : ℕ :=
  let f := s + 38  -- father's age
  let b := s - 4   -- brother's age
  let m := b + 36  -- mother's age
  f - m

/-- Theorem stating that the age difference between Sobha's parents is 6 years -/
theorem parents_age_difference (s : ℕ) (h : s ≥ 4) : age_difference s = 6 := by
  sorry

end NUMINAMATH_CALUDE_parents_age_difference_l2274_227431


namespace NUMINAMATH_CALUDE_mixed_fractions_sum_product_l2274_227496

theorem mixed_fractions_sum_product : 
  (9 + 1/2 + 7 + 1/6 + 5 + 1/12 + 3 + 1/20 + 1 + 1/30) * 12 = 310 := by
  sorry

end NUMINAMATH_CALUDE_mixed_fractions_sum_product_l2274_227496


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2274_227483

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2274_227483


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2274_227436

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - i) / (2 + 5*i) = (1:ℂ) / 29 - (17:ℂ) / 29 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2274_227436


namespace NUMINAMATH_CALUDE_sector_angle_and_area_l2274_227470

/-- Given a sector with radius 8 and arc length 12, prove its central angle and area -/
theorem sector_angle_and_area :
  let r : ℝ := 8
  let l : ℝ := 12
  let α : ℝ := l / r
  let S : ℝ := (1 / 2) * l * r
  α = 3 / 2 ∧ S = 48 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_and_area_l2274_227470


namespace NUMINAMATH_CALUDE_sum_first_105_remainder_l2274_227434

theorem sum_first_105_remainder (n : Nat) (d : Nat) : n = 105 → d = 5270 → (n * (n + 1) / 2) % d = 295 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_105_remainder_l2274_227434


namespace NUMINAMATH_CALUDE_train_speed_problem_l2274_227413

/-- Given a train journey with the following properties:
  * Total distance is 3x km
  * First part of the journey covers x km at speed V kmph
  * Second part of the journey covers 2x km at 20 kmph
  * Average speed for the entire journey is 27 kmph
  Then, the speed V of the first part of the journey is 90 kmph. -/
theorem train_speed_problem (x : ℝ) (V : ℝ) (h_x_pos : x > 0) (h_V_pos : V > 0) :
  (x / V + 2 * x / 20) = 3 * x / 27 → V = 90 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2274_227413


namespace NUMINAMATH_CALUDE_maurice_cookout_beef_per_package_l2274_227484

/-- Calculates the amount of ground beef per package for Maurice's cookout -/
theorem maurice_cookout_beef_per_package 
  (total_people : ℕ) 
  (beef_per_person : ℕ) 
  (num_packages : ℕ) 
  (h1 : total_people = 10) 
  (h2 : beef_per_person = 2) 
  (h3 : num_packages = 4) : 
  (total_people * beef_per_person) / num_packages = 5 := by
  sorry

#check maurice_cookout_beef_per_package

end NUMINAMATH_CALUDE_maurice_cookout_beef_per_package_l2274_227484


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2274_227420

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2274_227420


namespace NUMINAMATH_CALUDE_tampa_bay_bucs_problem_l2274_227458

/-- The Tampa Bay Bucs team composition problem -/
theorem tampa_bay_bucs_problem 
  (initial_football_players : ℕ)
  (initial_cheerleaders : ℕ)
  (quitting_football_players : ℕ)
  (quitting_cheerleaders : ℕ)
  (h1 : initial_football_players = 13)
  (h2 : initial_cheerleaders = 16)
  (h3 : quitting_football_players = 10)
  (h4 : quitting_cheerleaders = 4) :
  (initial_football_players - quitting_football_players) + 
  (initial_cheerleaders - quitting_cheerleaders) = 15 := by
  sorry

end NUMINAMATH_CALUDE_tampa_bay_bucs_problem_l2274_227458


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l2274_227448

-- Define the hyperbola equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) - y^2 / (m - 1) = 1

-- Define the condition that foci are on x-axis
def foci_on_x_axis (m : ℝ) : Prop :=
  m + 2 > 0 ∧ m - 1 > 0

-- Theorem statement
theorem hyperbola_m_range (m : ℝ) :
  is_hyperbola m ∧ foci_on_x_axis m → m > 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l2274_227448


namespace NUMINAMATH_CALUDE_strawberry_basket_count_l2274_227409

theorem strawberry_basket_count (baskets : ℕ) (friends : ℕ) (total : ℕ) :
  baskets = 6 →
  friends = 3 →
  total = 1200 →
  ∃ (strawberries_per_basket : ℕ),
    strawberries_per_basket * baskets * (friends + 1) = total ∧
    strawberries_per_basket = 50 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_basket_count_l2274_227409


namespace NUMINAMATH_CALUDE_disjunction_is_true_l2274_227479

def p : Prop := 1 ∈ {x : ℝ | (x + 2) * (x - 3) < 0}

def q : Prop := (∅ : Set ℕ) = {0}

theorem disjunction_is_true : p ∨ q := by sorry

end NUMINAMATH_CALUDE_disjunction_is_true_l2274_227479


namespace NUMINAMATH_CALUDE_right_triangle_in_square_l2274_227447

theorem right_triangle_in_square (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ (s : ℝ), s > 0 ∧ s^2 = 16 ∧ a^2 + b^2 = s^2) →
  a * b = 16 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_in_square_l2274_227447


namespace NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2274_227493

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- Diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating the diameter of the bacterium in nanometers -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2.85 * 10^3 := by
  sorry

#check bacterium_diameter_nanometers

end NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2274_227493


namespace NUMINAMATH_CALUDE_sin_180_degrees_l2274_227495

theorem sin_180_degrees : Real.sin (π) = 0 := by sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l2274_227495


namespace NUMINAMATH_CALUDE_x0_range_l2274_227404

/-- Circle C with equation x^2 + y^2 = 1 -/
def Circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

/-- Line l with equation 3x + 2y - 4 = 0 -/
def Line_l : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 + 2 * p.2 - 4 = 0}

/-- Condition that there always exist two different points A, B on circle C such that OA + OB = OP -/
def ExistPoints (P : ℝ × ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ Circle_C → B ∈ Circle_C → A ≠ B → 
    (A.1, A.2) + (B.1, B.2) = P

theorem x0_range (x0 y0 : ℝ) (hP : (x0, y0) ∈ Line_l) 
    (hExist : ExistPoints (x0, y0)) : 
  0 < x0 ∧ x0 < 24/13 := by
  sorry

end NUMINAMATH_CALUDE_x0_range_l2274_227404


namespace NUMINAMATH_CALUDE_total_miles_run_l2274_227446

theorem total_miles_run (xavier katie cole lily joe : ℝ) : 
  xavier = 3 * katie → 
  katie = 4 * cole → 
  lily = 5 * cole → 
  joe = 2 * lily → 
  xavier = 84 → 
  lily = 0.85 * joe → 
  xavier + katie + cole + lily + joe = 168.875 := by
sorry

end NUMINAMATH_CALUDE_total_miles_run_l2274_227446


namespace NUMINAMATH_CALUDE_exponential_inequality_l2274_227422

theorem exponential_inequality (m : ℝ) (h : 0 < m ∧ m < 1) :
  (1 - m) ^ (1/3 : ℝ) > (1 - m) ^ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2274_227422


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2274_227460

theorem radical_conjugate_sum_product (x y : ℝ) : 
  (x + Real.sqrt y) + (x - Real.sqrt y) = 10 →
  (x + Real.sqrt y) * (x - Real.sqrt y) = 9 →
  x + y = 21 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l2274_227460


namespace NUMINAMATH_CALUDE_max_gcd_of_product_7200_l2274_227489

theorem max_gcd_of_product_7200 :
  ∃ (a b : ℕ), a * b = 7200 ∧ 
  ∀ (c d : ℕ), c * d = 7200 → Nat.gcd c d ≤ 60 ∧
  Nat.gcd a b = 60 :=
sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_7200_l2274_227489


namespace NUMINAMATH_CALUDE_hair_cut_length_l2274_227405

/-- The amount of hair cut off is equal to the difference between the initial hair length and the final hair length. -/
theorem hair_cut_length (initial_length final_length cut_length : ℕ) 
  (h1 : initial_length = 18)
  (h2 : final_length = 9)
  (h3 : cut_length = initial_length - final_length) :
  cut_length = 9 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l2274_227405


namespace NUMINAMATH_CALUDE_inequality_proof_l2274_227442

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b * c / a + c * a / b + a * b / c ≥ a + b + c) ∧
  (a + b + c = 1 → (1 - a) / a + (1 - b) / b + (1 - c) / c ≥ 6) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2274_227442


namespace NUMINAMATH_CALUDE_min_value_theorem_l2274_227410

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line equation -/
def line_eq (a b x y : ℝ) : Prop := 2*a*x - b*y + 2 = 0

/-- The center of the circle satisfies the circle equation -/
def center_satisfies_circle (x₀ y₀ : ℝ) : Prop := circle_eq x₀ y₀

/-- The line passes through the center of the circle -/
def line_passes_through_center (a b x₀ y₀ : ℝ) : Prop := line_eq a b x₀ y₀

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (x₀ y₀ : ℝ) (h_center : center_satisfies_circle x₀ y₀) 
  (h_line : line_passes_through_center a b x₀ y₀) : 
  (1 / a + 1 / b) ≥ 4 ∧ ∃ (a₀ b₀ : ℝ), 1 / a₀ + 1 / b₀ = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2274_227410


namespace NUMINAMATH_CALUDE_linear_approximation_of_f_l2274_227424

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 5)

theorem linear_approximation_of_f :
  let a : ℝ := 2
  let x : ℝ := 1.97
  let f_a : ℝ := f a
  let f'_a : ℝ := a / Real.sqrt (a^2 + 5)
  let Δx : ℝ := x - a
  let approximation : ℝ := f_a + f'_a * Δx
  ∃ ε > 0, |approximation - 2.98| < ε :=
by
  sorry

#check linear_approximation_of_f

end NUMINAMATH_CALUDE_linear_approximation_of_f_l2274_227424


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l2274_227440

theorem cubic_polynomial_roots :
  let p (x : ℝ) := x^3 - 2*x^2 - 5*x + 6
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l2274_227440


namespace NUMINAMATH_CALUDE_small_bottles_sold_percentage_l2274_227412

theorem small_bottles_sold_percentage 
  (initial_small : ℕ) 
  (initial_big : ℕ) 
  (big_sold_percent : ℚ) 
  (total_remaining : ℕ) :
  initial_small = 6000 →
  initial_big = 10000 →
  big_sold_percent = 15/100 →
  total_remaining = 13780 →
  ∃ (small_sold_percent : ℚ),
    small_sold_percent = 12/100 ∧
    (initial_small * (1 - small_sold_percent)).floor + 
    (initial_big * (1 - big_sold_percent)).floor = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_small_bottles_sold_percentage_l2274_227412


namespace NUMINAMATH_CALUDE_sin_585_degrees_l2274_227429

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l2274_227429


namespace NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_97_l2274_227411

theorem first_nonzero_digit_after_decimal_1_97 : ∃ (n : ℕ) (d : ℕ), 
  0 < d ∧ d < 10 ∧ 
  (∃ (k : ℕ), 10^n ≤ k * 97 ∧ k * 97 < 10^(n+1) ∧ 
  (10^(n+1) * 1 - k * 97) / 97 = d) ∧
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_after_decimal_1_97_l2274_227411


namespace NUMINAMATH_CALUDE_complex_fraction_equation_solution_l2274_227477

theorem complex_fraction_equation_solution :
  ∃ x : ℚ, 3 + 1 / (2 + 1 / (3 + 3 / (4 + x))) = 225/68 ∧ x = -50/19 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equation_solution_l2274_227477


namespace NUMINAMATH_CALUDE_x_one_value_l2274_227476

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/4) : 
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l2274_227476


namespace NUMINAMATH_CALUDE_total_books_eq_sum_l2274_227455

/-- The total number of different books in the 'crazy silly school' series -/
def total_books : ℕ := sorry

/-- The number of books already read from the series -/
def books_read : ℕ := 8

/-- The number of books left to read from the series -/
def books_left : ℕ := 6

/-- Theorem stating that the total number of books is equal to the sum of books read and books left to read -/
theorem total_books_eq_sum : total_books = books_read + books_left := by sorry

end NUMINAMATH_CALUDE_total_books_eq_sum_l2274_227455


namespace NUMINAMATH_CALUDE_weight_difference_l2274_227492

/-- Given the weights of four individuals with specific relationships, prove the weight difference between two of them. -/
theorem weight_difference (total_weight : ℝ) (jack_weight : ℝ) (avg_weight : ℝ) : 
  total_weight = 240 ∧ 
  jack_weight = 52 ∧ 
  avg_weight = 60 →
  ∃ (sam_weight lisa_weight daisy_weight : ℝ),
    sam_weight = jack_weight / 0.8 ∧
    lisa_weight = jack_weight * 1.4 ∧
    daisy_weight = (jack_weight + lisa_weight) / 3 ∧
    total_weight = jack_weight + sam_weight + lisa_weight + daisy_weight ∧
    sam_weight - daisy_weight = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l2274_227492


namespace NUMINAMATH_CALUDE_perfect_squares_mod_six_l2274_227437

theorem perfect_squares_mod_six :
  (∀ n : ℤ, n^2 % 6 ≠ 2) ∧
  (∃ K : Set ℤ, Set.Infinite K ∧ ∀ k ∈ K, ((6 * k + 3)^2) % 6 = 3) :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_mod_six_l2274_227437


namespace NUMINAMATH_CALUDE_wall_area_calculation_l2274_227414

theorem wall_area_calculation (regular_area : ℝ) (jumbo_ratio : ℝ) 
  (regular_length : ℝ) (regular_width : ℝ) (jumbo_length : ℝ) :
  regular_area = 90 →
  jumbo_ratio = 1 / 3 →
  jumbo_length = 3 * regular_length →
  regular_length / regular_width = jumbo_length / regular_width →
  (regular_area + jumbo_ratio / (1 - jumbo_ratio) * regular_area * (jumbo_length / regular_length)^2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_wall_area_calculation_l2274_227414


namespace NUMINAMATH_CALUDE_final_bacteria_count_l2274_227400

def initial_count : ℕ := 30
def start_time : ℕ := 0  -- 10:00 AM represented as 0 minutes
def end_time : ℕ := 30   -- 10:30 AM represented as 30 minutes
def growth_interval : ℕ := 5  -- population triples every 5 minutes
def death_interval : ℕ := 15  -- 10% die every 15 minutes

def growth_factor : ℚ := 3
def survival_rate : ℚ := 0.9  -- 90% survival rate (10% die)

def number_of_growth_periods (t : ℕ) : ℕ := t / growth_interval

def number_of_death_periods (t : ℕ) : ℕ := t / death_interval

def bacteria_count (t : ℕ) : ℚ :=
  initial_count *
  growth_factor ^ (number_of_growth_periods t) *
  survival_rate ^ (number_of_death_periods t)

theorem final_bacteria_count :
  bacteria_count end_time = 17694 := by sorry

end NUMINAMATH_CALUDE_final_bacteria_count_l2274_227400


namespace NUMINAMATH_CALUDE_find_a_l2274_227430

-- Define the universal set U
def U (a : ℝ) : Set ℝ := {2, a^2 + 9*a + 3, 6}

-- Define set A
def A (a : ℝ) : Set ℝ := {2, |a + 3|}

-- Define the complement of A relative to U
def complement_A (a : ℝ) : Set ℝ := {3}

-- Theorem statement
theorem find_a : ∃ a : ℝ, 
  (U a = {2, a^2 + 9*a + 3, 6}) ∧ 
  (A a = {2, |a + 3|}) ∧ 
  (complement_A a = {3}) ∧ 
  (a = -9) := by
  sorry

end NUMINAMATH_CALUDE_find_a_l2274_227430


namespace NUMINAMATH_CALUDE_set_four_subsets_implies_a_not_zero_or_two_l2274_227425

theorem set_four_subsets_implies_a_not_zero_or_two (a : ℝ) : 
  (Finset.powerset {a, a^2 - a}).card = 4 → a ≠ 0 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_set_four_subsets_implies_a_not_zero_or_two_l2274_227425


namespace NUMINAMATH_CALUDE_sum_of_digits_l2274_227406

/-- 
Given a three-digit number ABC, where:
- ABC is an integer between 100 and 999 (inclusive)
- ABC = 17 * 28 + 9

Prove that the sum of its digits A, B, and C is 17.
-/
theorem sum_of_digits (ABC : ℕ) (h1 : 100 ≤ ABC) (h2 : ABC ≤ 999) (h3 : ABC = 17 * 28 + 9) :
  (ABC / 100) + ((ABC / 10) % 10) + (ABC % 10) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2274_227406


namespace NUMINAMATH_CALUDE_average_rate_of_change_f_l2274_227402

def f (x : ℝ) := x^2 - 1

theorem average_rate_of_change_f : 
  let x₁ : ℝ := 1
  let x₂ : ℝ := 1.1
  (f x₂ - f x₁) / (x₂ - x₁) = 2.1 := by
sorry

end NUMINAMATH_CALUDE_average_rate_of_change_f_l2274_227402


namespace NUMINAMATH_CALUDE_compare_towers_two_three_compare_towers_three_four_l2274_227401

def tower_of_two (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => 2^(tower_of_two n)

def tower_of_three (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => 3^(tower_of_three n)

def tower_of_four (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n+1 => 4^(tower_of_four n)

theorem compare_towers_two_three (n : ℕ) (h : n ≥ 3) : 
  tower_of_two n < tower_of_three (n-1) :=
sorry

theorem compare_towers_three_four (n : ℕ) (h : n ≥ 2) : 
  tower_of_three n > tower_of_four (n-1) :=
sorry

end NUMINAMATH_CALUDE_compare_towers_two_three_compare_towers_three_four_l2274_227401


namespace NUMINAMATH_CALUDE_multiply_24_99_l2274_227463

theorem multiply_24_99 : 24 * 99 = 2376 := by
  sorry

end NUMINAMATH_CALUDE_multiply_24_99_l2274_227463


namespace NUMINAMATH_CALUDE_initial_player_count_l2274_227471

/-- Represents a server in the Minecraft scenario -/
structure Server :=
  (players : ℕ)

/-- Represents the state of the two servers at a given time -/
structure GameState :=
  (server1 : Server)
  (server2 : Server)

/-- Simulates a single step of the game, where a player may switch servers -/
def step (state : GameState) : GameState :=
  if state.server1.players > state.server2.players
  then { server1 := ⟨state.server1.players - 1⟩, server2 := ⟨state.server2.players + 1⟩ }
  else if state.server2.players > state.server1.players
  then { server1 := ⟨state.server1.players + 1⟩, server2 := ⟨state.server2.players - 1⟩ }
  else state

/-- Simulates the entire game for a given number of steps -/
def simulate (initial : GameState) (steps : ℕ) : GameState :=
  match steps with
  | 0 => initial
  | n + 1 => step (simulate initial n)

/-- The theorem stating the possible initial player counts -/
theorem initial_player_count (initial : GameState) :
  (simulate initial 2023).server1.players + (simulate initial 2023).server2.players = initial.server1.players + initial.server2.players →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server1.players ≠ 0) →
  (∀ i : ℕ, i ≤ 2023 → (simulate initial i).server2.players ≠ 0) →
  initial.server1.players = 1011 ∨ initial.server1.players = 1012 :=
sorry

end NUMINAMATH_CALUDE_initial_player_count_l2274_227471


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2274_227421

/-- A regular polygon with interior angles measuring 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = (150 * n : ℝ) →
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2274_227421


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2274_227494

/-- The eccentricity of a hyperbola with equation x^2/2 - y^2 = 1 is √6/2 -/
theorem hyperbola_eccentricity :
  let a : ℝ := Real.sqrt 2
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 3
  let e : ℝ := c / a
  e = Real.sqrt 6 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2274_227494


namespace NUMINAMATH_CALUDE_coefficient_of_y_l2274_227466

theorem coefficient_of_y (x y : ℝ) (a : ℝ) : 
  x / (2 * y) = 3 / 2 → 
  (7 * x + a * y) / (x - 2 * y) = 26 → 
  a = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_y_l2274_227466


namespace NUMINAMATH_CALUDE_tomato_count_l2274_227474

/-- Represents a rectangular garden with tomatoes -/
structure TomatoGarden where
  rows : ℕ
  columns : ℕ
  tomato_position : ℕ × ℕ

/-- Calculates the total number of tomatoes in the garden -/
def total_tomatoes (garden : TomatoGarden) : ℕ :=
  garden.rows * garden.columns

/-- Theorem stating the total number of tomatoes in the garden -/
theorem tomato_count (garden : TomatoGarden) 
  (h1 : garden.tomato_position.1 = 8)  -- 8th row from front
  (h2 : garden.rows - garden.tomato_position.1 + 1 = 14)  -- 14th row from back
  (h3 : garden.tomato_position.2 = 7)  -- 7th row from left
  (h4 : garden.columns - garden.tomato_position.2 + 1 = 13)  -- 13th row from right
  : total_tomatoes garden = 399 := by
  sorry

#eval total_tomatoes { rows := 21, columns := 19, tomato_position := (8, 7) }

end NUMINAMATH_CALUDE_tomato_count_l2274_227474


namespace NUMINAMATH_CALUDE_exists_year_with_special_form_l2274_227423

def is_21st_century (y : ℕ) : Prop := 2001 ≤ y ∧ y ≤ 2100

def are_distinct_digits (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem exists_year_with_special_form :
  ∃ (y : ℕ) (a b c d e f g h i j : ℕ),
    is_21st_century y ∧
    are_distinct_digits a b c d e f g h i j ∧
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧ is_digit e ∧
    is_digit f ∧ is_digit g ∧ is_digit h ∧ is_digit i ∧ is_digit j ∧
    y = (a + b * c * d * e) / (f + g * h * i * j) :=
sorry

end NUMINAMATH_CALUDE_exists_year_with_special_form_l2274_227423


namespace NUMINAMATH_CALUDE_one_painted_face_probability_l2274_227475

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  painted_faces : ℕ
  painted_faces_adjacent : Bool

/-- Calculates the number of unit cubes with exactly one painted face -/
def one_painted_face_count (c : PaintedCube) : ℕ :=
  if c.painted_faces_adjacent then
    2 * (c.side_length^2 - c.side_length) - (c.side_length - 1)
  else
    c.painted_faces * (c.side_length^2 - c.side_length)

/-- Theorem stating the probability of selecting a unit cube with one painted face -/
theorem one_painted_face_probability (c : PaintedCube) 
  (h1 : c.side_length = 5)
  (h2 : c.painted_faces = 2)
  (h3 : c.painted_faces_adjacent = true) :
  (one_painted_face_count c : ℚ) / (c.side_length^3 : ℚ) = 41 / 125 := by
  sorry

end NUMINAMATH_CALUDE_one_painted_face_probability_l2274_227475


namespace NUMINAMATH_CALUDE_triangle_translation_l2274_227418

structure Point where
  x : ℝ
  y : ℝ

def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem triangle_translation :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨0, 2⟩
  let A' : Point := ⟨-1, 5⟩
  let dx : ℝ := A'.x - A.x
  let dy : ℝ := A'.y - A.y
  let C' : Point := translate C dx dy
  C'.x = -3 ∧ C'.y = 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_translation_l2274_227418


namespace NUMINAMATH_CALUDE_sum_of_solutions_l2274_227444

theorem sum_of_solutions (x : ℝ) : (x + 16 / x = 12) → (∃ y : ℝ, y + 16 / y = 12 ∧ y ≠ x) → x + y = 12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l2274_227444


namespace NUMINAMATH_CALUDE_no_real_solutions_to_inequality_l2274_227459

theorem no_real_solutions_to_inequality :
  ¬∃ x : ℝ, x ≠ 5 ∧ (x^3 - 125) / (x - 5) < 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_to_inequality_l2274_227459


namespace NUMINAMATH_CALUDE_line_bisects_circle_l2274_227467

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem line_bisects_circle :
  line_eq (circle_center.1) (circle_center.2) ∧
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_eq x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_line_bisects_circle_l2274_227467


namespace NUMINAMATH_CALUDE_gallery_to_work_blocks_l2274_227456

/-- The number of blocks from start to work -/
def total_blocks : ℕ := 37

/-- The number of blocks to the store -/
def store_blocks : ℕ := 11

/-- The number of blocks to the gallery -/
def gallery_blocks : ℕ := 6

/-- The number of blocks already walked -/
def walked_blocks : ℕ := 5

/-- The number of remaining blocks to work after walking 5 blocks -/
def remaining_blocks : ℕ := 20

/-- The number of blocks from the gallery to work -/
def gallery_to_work : ℕ := total_blocks - walked_blocks - store_blocks - gallery_blocks

theorem gallery_to_work_blocks :
  gallery_to_work = 15 :=
by sorry

end NUMINAMATH_CALUDE_gallery_to_work_blocks_l2274_227456


namespace NUMINAMATH_CALUDE_mall_entrance_exit_ways_l2274_227462

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

end NUMINAMATH_CALUDE_mall_entrance_exit_ways_l2274_227462


namespace NUMINAMATH_CALUDE_only_negative_number_l2274_227465

theorem only_negative_number (a b c d : ℝ) : 
  a = 2023 → b = -2023 → c = 1 / 2023 → d = 0 →
  (b < 0 ∧ a > 0 ∧ c > 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_number_l2274_227465


namespace NUMINAMATH_CALUDE_circle_line_intersection_theorem_l2274_227488

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ

/-- Represents a line in the 2D plane -/
structure Line where

/-- The length of the chord formed by the intersection of a circle and a line -/
def chordLength (c : Circle) (l : Line) : ℝ := 4

/-- Theorem: If the chord length is 4, then the value of 'a' in the circle equation is -4 -/
theorem circle_line_intersection_theorem (c : Circle) (l : Line) : 
  chordLength c l = 4 → c.a = -4 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_theorem_l2274_227488


namespace NUMINAMATH_CALUDE_division_problem_l2274_227432

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 136 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder → 
  divisor = 15 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2274_227432


namespace NUMINAMATH_CALUDE_biggest_number_l2274_227407

theorem biggest_number (jungkook yoongi yuna : ℚ) : 
  jungkook = 6 / 3 → yoongi = 4 → yuna = 5 → 
  max (max jungkook yoongi) yuna = 5 := by
sorry

end NUMINAMATH_CALUDE_biggest_number_l2274_227407


namespace NUMINAMATH_CALUDE_unique_solution_cube_sum_l2274_227451

theorem unique_solution_cube_sum (n : ℕ+) : 
  (∃ (x y z : ℕ+), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2) ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_sum_l2274_227451


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l2274_227454

theorem lcm_gcd_product (a b : ℕ) (h1 : a = 24) (h2 : b = 54) :
  (Nat.lcm a b) * (Nat.gcd a b) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l2274_227454


namespace NUMINAMATH_CALUDE_max_distinct_dance_counts_l2274_227498

/-- Represents the dance count for a person -/
def DanceCount := Nat

/-- Represents a set of distinct dance counts -/
def DistinctCounts := Finset DanceCount

theorem max_distinct_dance_counts 
  (num_boys : Nat) 
  (num_girls : Nat) 
  (h_boys : num_boys = 29) 
  (h_girls : num_girls = 15) :
  ∃ (dc : DistinctCounts), dc.card ≤ 29 ∧ 
  ∀ (dc' : DistinctCounts), dc'.card ≤ dc.card :=
sorry

end NUMINAMATH_CALUDE_max_distinct_dance_counts_l2274_227498


namespace NUMINAMATH_CALUDE_reflected_rays_angle_l2274_227427

theorem reflected_rays_angle 
  (α β : Real) 
  (h_α : 0 < α ∧ α < π/2) 
  (h_β : 0 < β ∧ β < π/2) : 
  ∃ θ : Real, θ = Real.arccos (1 - 2 * Real.sin α ^ 2 * Real.sin β ^ 2) := by
sorry

end NUMINAMATH_CALUDE_reflected_rays_angle_l2274_227427


namespace NUMINAMATH_CALUDE_not_counterexample_58_l2274_227449

def is_counterexample (n : ℕ) : Prop :=
  Nat.Prime n ∧ ¬(Nat.Prime (n + 2))

theorem not_counterexample_58 : ¬(is_counterexample 58) := by
  sorry

end NUMINAMATH_CALUDE_not_counterexample_58_l2274_227449


namespace NUMINAMATH_CALUDE_irreducible_fraction_l2274_227468

theorem irreducible_fraction (n : ℕ) :
  (Nat.gcd (n^3 + n) (2*n + 1) = 1) ↔ (n % 5 ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_irreducible_fraction_l2274_227468


namespace NUMINAMATH_CALUDE_sport_formulation_corn_syrup_l2274_227417

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio where
  flavoring : ℚ
  corn_syrup : ℚ
  water : ℚ

/-- The standard formulation ratio -/
def standard_ratio : DrinkRatio := ⟨1, 12, 30⟩

/-- The sport formulation ratio -/
def sport_ratio (standard : DrinkRatio) : DrinkRatio :=
  ⟨standard.flavoring,
   standard.corn_syrup / 3,
   standard.water * 2⟩

/-- Calculates the amount of corn syrup given the amount of water and the ratio -/
def corn_syrup_amount (water_amount : ℚ) (ratio : DrinkRatio) : ℚ :=
  (ratio.corn_syrup * water_amount) / ratio.water

theorem sport_formulation_corn_syrup :
  corn_syrup_amount 30 (sport_ratio standard_ratio) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sport_formulation_corn_syrup_l2274_227417


namespace NUMINAMATH_CALUDE_age_ratio_ten_years_ago_l2274_227469

-- Define Alice's current age
def alice_current_age : ℕ := 30

-- Define the age difference between Alice and Tom
def age_difference : ℕ := 15

-- Define the number of years that have passed
def years_passed : ℕ := 10

-- Define Tom's current age
def tom_current_age : ℕ := alice_current_age - age_difference

-- Define Alice's age 10 years ago
def alice_past_age : ℕ := alice_current_age - years_passed

-- Define Tom's age 10 years ago
def tom_past_age : ℕ := tom_current_age - years_passed

-- Theorem to prove
theorem age_ratio_ten_years_ago :
  alice_past_age / tom_past_age = 4 :=
sorry

end NUMINAMATH_CALUDE_age_ratio_ten_years_ago_l2274_227469


namespace NUMINAMATH_CALUDE_reunion_boys_l2274_227464

/-- The number of handshakes when n people each shake hands with everyone else exactly once -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- There were 8 boys at the reunion -/
theorem reunion_boys : ∃ n : ℕ, n > 0 ∧ handshakes n = 28 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_reunion_boys_l2274_227464


namespace NUMINAMATH_CALUDE_rachels_weight_l2274_227403

theorem rachels_weight (rachel jimmy adam : ℝ) 
  (h1 : jimmy = rachel + 6)
  (h2 : rachel = adam + 15)
  (h3 : (rachel + jimmy + adam) / 3 = 72) :
  rachel = 75 := by
  sorry

end NUMINAMATH_CALUDE_rachels_weight_l2274_227403


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_12_l2274_227490

theorem rationalize_denominator_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_12_l2274_227490


namespace NUMINAMATH_CALUDE_quadratic_roots_interlace_l2274_227450

/-- 
Given real numbers p₁, p₂, q₁, q₂ satisfying the inequality
(q₁ - q₂)² + (p₁ - p₂)(p₁q₂ - p₂q₁) < 0,
prove that the quadratic polynomials x^2 + p₁x + q₁ and x^2 + p₂x + q₂
each have two real roots, and between the two roots of each polynomial
lies a root of the other polynomial.
-/
theorem quadratic_roots_interlace (p₁ p₂ q₁ q₂ : ℝ) 
  (h : (q₁ - q₂)^2 + (p₁ - p₂) * (p₁ * q₂ - p₂ * q₁) < 0) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ < x₃ ∧ x₃ < x₂) ∧ 
    (x₃ < x₄ ∧ x₄ < x₂) ∧
    (x^2 + p₁*x + q₁ = 0 ↔ x = x₁ ∨ x = x₂) ∧
    (x^2 + p₂*x + q₂ = 0 ↔ x = x₃ ∨ x = x₄) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_interlace_l2274_227450


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_in_reflected_triangle_l2274_227491

/-- Represents a rectangle formed by reflecting an isosceles triangle over its base -/
structure ReflectedTriangleRectangle where
  base : ℝ
  height : ℝ
  inscribed_semicircle_radius : ℝ

/-- The theorem stating the radius of the inscribed semicircle in the specific rectangle -/
theorem inscribed_semicircle_radius_in_reflected_triangle
  (rect : ReflectedTriangleRectangle)
  (h_base : rect.base = 24)
  (h_height : rect.height = 10) :
  rect.inscribed_semicircle_radius = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_in_reflected_triangle_l2274_227491


namespace NUMINAMATH_CALUDE_seven_double_prime_l2274_227435

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem seven_double_prime : prime (prime 7) = 51 := by
  sorry

end NUMINAMATH_CALUDE_seven_double_prime_l2274_227435


namespace NUMINAMATH_CALUDE_scaled_prism_marbles_l2274_227480

/-- Represents a triangular prism-shaped container -/
structure TriangularPrism where
  baseArea : ℝ
  height : ℝ
  marbles : ℕ

/-- Scales the dimensions of a triangular prism by a given factor -/
def scalePrism (p : TriangularPrism) (factor : ℝ) : TriangularPrism :=
  { baseArea := p.baseArea * factor^2
  , height := p.height * factor
  , marbles := p.marbles }

/-- Theorem: Scaling a triangular prism by a factor of 2 results in 8 times the marbles -/
theorem scaled_prism_marbles (p : TriangularPrism) :
  (scalePrism p 2).marbles = 8 * p.marbles :=
by sorry

end NUMINAMATH_CALUDE_scaled_prism_marbles_l2274_227480


namespace NUMINAMATH_CALUDE_fifteen_valid_pairs_l2274_227478

/-- A function that constructs the number 7ABABA from single digits A and B -/
def constructNumber (A B : Nat) : Nat :=
  700000 + 10000 * A + 1000 * B + 100 * A + 10 * B + A

/-- Predicate to check if a number is a single digit -/
def isSingleDigit (n : Nat) : Prop := n < 10

/-- The main theorem stating that there are exactly 15 valid pairs (A, B) -/
theorem fifteen_valid_pairs :
  ∃! (validPairs : Finset (Nat × Nat)),
    validPairs.card = 15 ∧
    ∀ (A B : Nat),
      (A, B) ∈ validPairs ↔
        isSingleDigit A ∧
        isSingleDigit B ∧
        (constructNumber A B % 6 = 0) :=
  sorry

end NUMINAMATH_CALUDE_fifteen_valid_pairs_l2274_227478


namespace NUMINAMATH_CALUDE_sugar_consumption_change_l2274_227415

theorem sugar_consumption_change 
  (original_price : ℝ) 
  (original_consumption : ℝ) 
  (price_increase_percentage : ℝ) 
  (expenditure_increase_percentage : ℝ) 
  (h1 : original_consumption = 30)
  (h2 : price_increase_percentage = 0.32)
  (h3 : expenditure_increase_percentage = 0.10) : 
  ∃ new_consumption : ℝ, 
    new_consumption = 25 ∧ 
    (1 + expenditure_increase_percentage) * (original_consumption * original_price) = 
    new_consumption * ((1 + price_increase_percentage) * original_price) :=
by sorry

end NUMINAMATH_CALUDE_sugar_consumption_change_l2274_227415


namespace NUMINAMATH_CALUDE_f_properties_l2274_227481

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem f_properties :
  (¬ (∀ x, f (-x) = -f x) ∧ ¬ (∀ x, f (-x) = f x)) ∧
  (∃ y, f 1 ≤ f y ∧ ∀ x, f 1 ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2274_227481


namespace NUMINAMATH_CALUDE_exponent_equation_l2274_227426

theorem exponent_equation (a b : ℤ) : 3^a * 9^b = (1:ℚ)/3 → a + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equation_l2274_227426


namespace NUMINAMATH_CALUDE_car_distance_calculation_l2274_227439

/-- Proves that the distance covered by a car traveling at 99 km/h for 5 hours is 495 km -/
theorem car_distance_calculation (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 99)
  (h2 : time = 5)
  (h3 : distance = speed * time) : 
  distance = 495 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l2274_227439


namespace NUMINAMATH_CALUDE_min_cost_2009_l2274_227416

/-- Represents the denominations of coins available --/
inductive Coin
  | One
  | Two
  | Five
  | Ten

/-- Represents an arithmetic expression --/
inductive Expr
  | Const (n : ℕ)
  | Add (e1 e2 : Expr)
  | Sub (e1 e2 : Expr)
  | Mul (e1 e2 : Expr)
  | Div (e1 e2 : Expr)

/-- Evaluates an expression to a natural number --/
def eval : Expr → ℕ
  | Expr.Const n => n
  | Expr.Add e1 e2 => eval e1 + eval e2
  | Expr.Sub e1 e2 => eval e1 - eval e2
  | Expr.Mul e1 e2 => eval e1 * eval e2
  | Expr.Div e1 e2 => eval e1 / eval e2

/-- Calculates the cost of an expression in rubles --/
def cost : Expr → ℕ
  | Expr.Const n => n
  | Expr.Add e1 e2 => cost e1 + cost e2
  | Expr.Sub e1 e2 => cost e1 + cost e2
  | Expr.Mul e1 e2 => cost e1 + cost e2
  | Expr.Div e1 e2 => cost e1 + cost e2

/-- Theorem: The minimum cost to create an expression equal to 2009 is 23 rubles --/
theorem min_cost_2009 :
  ∃ (e : Expr), eval e = 2009 ∧ cost e = 23 ∧
  (∀ (e' : Expr), eval e' = 2009 → cost e' ≥ 23) :=
sorry


end NUMINAMATH_CALUDE_min_cost_2009_l2274_227416
