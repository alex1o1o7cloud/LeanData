import Mathlib

namespace min_value_quadratic_ratio_l2224_222495

theorem min_value_quadratic_ratio (a b : ℝ) (h1 : b > 0) 
  (h2 : b^2 - 4*a = 0) : 
  (∀ x : ℝ, (a*x^2 + b*x + 1) / b ≥ 2) ∧ 
  (∃ x : ℝ, (a*x^2 + b*x + 1) / b = 2) := by
  sorry

end min_value_quadratic_ratio_l2224_222495


namespace incorrect_statements_l2224_222480

theorem incorrect_statements : 
  let statement1 := (∃ a b : ℚ, a + b = 5 ∧ a + b = -3)
  let statement2 := (∀ x : ℝ, ∃ q : ℚ, x = q)
  let statement3 := (∀ x : ℝ, |x| > 0)
  let statement4 := (∀ x : ℝ, x * x = x → (x = 0 ∨ x = 1))
  let statement5 := (∀ a b : ℚ, a + b = 0 → (a > 0 ∨ b > 0))
  (¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 ∧ ¬statement5) := by sorry

end incorrect_statements_l2224_222480


namespace peters_horse_food_l2224_222467

/-- Calculates the total amount of food needed to feed horses for a given number of days. -/
def total_food_needed (num_horses : ℕ) (oats_per_feeding : ℕ) (oats_feedings_per_day : ℕ) 
                      (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_feeding * oats_feedings_per_day + grain_per_day) * num_days

/-- Proves that Peter needs 132 pounds of food to feed his horses for 3 days. -/
theorem peters_horse_food : total_food_needed 4 4 2 3 3 = 132 := by
  sorry

end peters_horse_food_l2224_222467


namespace complement_of_union_equals_open_interval_l2224_222413

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_of_union_equals_open_interval :
  (A ∪ B)ᶜ = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end complement_of_union_equals_open_interval_l2224_222413


namespace inequality_and_existence_l2224_222402

theorem inequality_and_existence : 
  (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ Real.sqrt 3 * (x*y + y*z + z*x)) ∧ 
  (∃ k : ℝ, k > Real.sqrt 3 ∧ (∀ x y z : ℝ, x^2 + 2*y^2 + 3*z^2 ≥ k * (x*y + y*z + z*x))) := by
  sorry

end inequality_and_existence_l2224_222402


namespace product_of_fractions_l2224_222454

theorem product_of_fractions : (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) * (7 / 8) = 3 / 8 := by
  sorry

end product_of_fractions_l2224_222454


namespace fraction_simplification_l2224_222440

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b - c * b) / (a * b - a^2) = (a^2 + a * b + b^2 + a * c) / (a * b) :=
sorry

end fraction_simplification_l2224_222440


namespace solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l2224_222420

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = {x : ℝ | 1/2 < x ∧ x < 5/2} :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_nonempty_solution (a : ℝ) :
  (∃ x : ℝ, f x < a) ↔ a > 1 :=
sorry

end solution_set_f_less_than_2_range_of_a_for_nonempty_solution_l2224_222420


namespace converse_proposition_l2224_222448

theorem converse_proposition (a : ℝ) : a > 2 → a^2 ≥ 4 := by
  sorry

end converse_proposition_l2224_222448


namespace sum_product_bounds_l2224_222401

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (min max : ℝ), min = -1/4 ∧ max = 1/2 ∧
  (xy + xz + yz ≥ min ∧ xy + xz + yz ≤ max) ∧
  ∀ t, min ≤ t ∧ t ≤ max → ∃ (a b c : ℝ), a + b + c = 1 ∧ ab + ac + bc = t :=
sorry

end sum_product_bounds_l2224_222401


namespace cubic_expansion_coefficients_l2224_222416

theorem cubic_expansion_coefficients (a b : ℤ) : 
  (3 * b + 3 * a^2 = 99) ∧ (3 * a * b^2 = 162) → (a = 6 ∧ b = -3) := by
  sorry

end cubic_expansion_coefficients_l2224_222416


namespace translation_problem_l2224_222484

def complex_translation (z w : ℂ) : ℂ := z + w

theorem translation_problem (t : ℂ → ℂ) (h : t (1 + 3*I) = 4 + 7*I) :
  t (2 + 6*I) = 5 + 10*I :=
by
  sorry

end translation_problem_l2224_222484


namespace peach_count_difference_l2224_222488

/-- The number of red peaches in the basket -/
def red_peaches : ℕ := 17

/-- The number of green peaches in the basket -/
def green_peaches : ℕ := 16

/-- The difference between the number of red peaches and green peaches -/
def peach_difference : ℕ := red_peaches - green_peaches

theorem peach_count_difference : peach_difference = 1 := by
  sorry

end peach_count_difference_l2224_222488


namespace bicycle_price_l2224_222466

theorem bicycle_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) :
  upfront_payment = 200 →
  upfront_percentage = 0.20 →
  upfront_payment = upfront_percentage * total_price →
  total_price = 1000 := by
sorry

end bicycle_price_l2224_222466


namespace pasture_rental_problem_l2224_222426

/-- The pasture rental problem -/
theorem pasture_rental_problem 
  (total_cost : ℕ) 
  (a_horses b_horses c_horses : ℕ) 
  (b_months c_months : ℕ) 
  (b_payment : ℕ) 
  (h_total_cost : total_cost = 870)
  (h_a_horses : a_horses = 12)
  (h_b_horses : b_horses = 16)
  (h_c_horses : c_horses = 18)
  (h_b_months : b_months = 9)
  (h_c_months : c_months = 6)
  (h_b_payment : b_payment = 360)
  : ∃ (a_months : ℕ), 
    a_horses * a_months * b_payment = b_horses * b_months * (total_cost - b_payment - c_horses * c_months * b_payment / (b_horses * b_months)) ∧ 
    a_months = 8 :=
by sorry

end pasture_rental_problem_l2224_222426


namespace eldoria_license_plates_l2224_222453

/-- The number of vowels available for the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The number of characters in a valid license plate. -/
def licensePlateLength : ℕ := 5

/-- Calculates the number of valid license plates in Eldoria. -/
def numValidLicensePlates : ℕ :=
  numVowels * numLetters * numDigits * numDigits * numDigits

/-- Theorem stating the number of valid license plates in Eldoria. -/
theorem eldoria_license_plates :
  numValidLicensePlates = 130000 := by
  sorry

end eldoria_license_plates_l2224_222453


namespace cube_root_abs_square_sum_l2224_222457

theorem cube_root_abs_square_sum : ∃ (x : ℝ), x^3 = -8 ∧ x + |(-6)| - 2^2 = 0 := by
  sorry

end cube_root_abs_square_sum_l2224_222457


namespace total_expenditure_nine_persons_l2224_222490

/-- Given 9 persons, where 8 spend 30 Rs each and the 9th spends 20 Rs more than the average,
    prove that the total expenditure is 292.5 Rs -/
theorem total_expenditure_nine_persons :
  let num_persons : ℕ := 9
  let num_regular_spenders : ℕ := 8
  let regular_expenditure : ℚ := 30
  let extra_expenditure : ℚ := 20
  let total_expenditure : ℚ := num_regular_spenders * regular_expenditure +
    (((num_regular_spenders * regular_expenditure) / num_persons) + extra_expenditure)
  total_expenditure = 292.5 := by
sorry

end total_expenditure_nine_persons_l2224_222490


namespace adjacent_pair_properties_l2224_222444

/-- Definition of "adjacent number pairs" -/
def adjacent_pair (m n : ℚ) : Prop :=
  m / 2 + n / 5 = (m + n) / 7

theorem adjacent_pair_properties :
  ∃ (m n : ℚ),
    /- Part 1 -/
    (adjacent_pair 2 n → n = -25 / 2) ∧
    /- Part 2① -/
    (adjacent_pair m n → m = -4 * n / 25) ∧
    /- Part 2② -/
    (adjacent_pair m n ∧ 25 * m + n = 6 → m = 8 / 25 ∧ n = -2) := by
  sorry

end adjacent_pair_properties_l2224_222444


namespace vector_magnitude_l2224_222411

theorem vector_magnitude (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (2, -1)
  (a.1 * b.1 + a.2 * b.2 = 5) →
  ((a.1 + b.1)^2 + (a.2 + b.2)^2 = 8^2) →
  (b.1^2 + b.2^2 = 7^2) :=
by
  sorry

end vector_magnitude_l2224_222411


namespace triangle_area_is_15_5_l2224_222491

/-- Triangle ABC inscribed in a rectangle --/
structure TriangleInRectangle where
  -- Rectangle dimensions
  width : ℝ
  height : ℝ
  -- Vertex positions
  a_height : ℝ
  b_distance : ℝ
  c_distance : ℝ
  -- Conditions
  width_positive : width > 0
  height_positive : height > 0
  a_height_valid : 0 < a_height ∧ a_height < height
  b_distance_valid : 0 < b_distance ∧ b_distance < width
  c_distance_valid : 0 < c_distance ∧ c_distance < height

/-- The area of triangle ABC --/
def triangleArea (t : TriangleInRectangle) : ℝ :=
  t.width * t.height - (0.5 * t.width * t.c_distance + 0.5 * (t.height - t.a_height) * t.width + 0.5 * t.b_distance * t.a_height)

/-- Theorem: The area of triangle ABC is 15.5 square units --/
theorem triangle_area_is_15_5 (t : TriangleInRectangle) 
    (h_width : t.width = 6)
    (h_height : t.height = 4)
    (h_a_height : t.a_height = 1)
    (h_b_distance : t.b_distance = 3)
    (h_c_distance : t.c_distance = 1) : 
  triangleArea t = 15.5 := by
  sorry


end triangle_area_is_15_5_l2224_222491


namespace linear_function_two_points_l2224_222403

/-- A linear function passing through exactly two of three given points -/
theorem linear_function_two_points :
  ∃ (f : ℝ → ℝ) (a b : ℝ), 
    (∀ x, f x = a * x + b) ∧ 
    (f 0 = 0 ∧ f 1 = 1 ∧ f 2 ≠ 0) :=
by sorry

end linear_function_two_points_l2224_222403


namespace arithmetic_and_geometric_sequences_l2224_222442

/-- Arithmetic sequence sum -/
def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + (n * (n - 1) * d) / 2

/-- Geometric sequence term -/
def geometric_term (a₁ : ℤ) (q : ℤ) (n : ℕ) : ℤ :=
  a₁ * q ^ (n - 1)

theorem arithmetic_and_geometric_sequences :
  (arithmetic_sum (-2) 4 8 = 96) ∧
  (geometric_term 1 3 7 = 729) := by
  sorry

end arithmetic_and_geometric_sequences_l2224_222442


namespace single_female_fraction_l2224_222452

theorem single_female_fraction (total : ℕ) (h1 : total > 0) :
  let male_percent : ℚ := 70 / 100
  let married_percent : ℚ := 30 / 100
  let male_married_fraction : ℚ := 1 / 7
  let male_count := (male_percent * total).floor
  let female_count := total - male_count
  let married_count := (married_percent * total).floor
  let male_married_count := (male_married_fraction * male_count).floor
  let female_married_count := married_count - male_married_count
  let single_female_count := female_count - female_married_count
  (single_female_count : ℚ) / female_count = 1 / 3 :=
by sorry

end single_female_fraction_l2224_222452


namespace exists_unique_N_l2224_222443

theorem exists_unique_N : ∃ N : ℤ, N = 1719 ∧
  ∀ a b : ℤ, (N / 2 - a = b - N / 2) →
    ((∃ m n : ℕ+, a = 19 * m + 85 * n) ∨ (∃ m n : ℕ+, b = 19 * m + 85 * n)) ∧
    ¬((∃ m n : ℕ+, a = 19 * m + 85 * n) ∧ (∃ m n : ℕ+, b = 19 * m + 85 * n)) :=
by sorry

end exists_unique_N_l2224_222443


namespace vector_magnitude_l2224_222459

/-- Given vectors a and b, where a is perpendicular to (2a - b), prove that the magnitude of b is 2√10 -/
theorem vector_magnitude (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.1 = -2)
  (h'' : a.1 * (2 * a.1 - b.1) + a.2 * (2 * a.2 - b.2) = 0) :
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 10 := by
  sorry

end vector_magnitude_l2224_222459


namespace quadratic_expression_evaluation_l2224_222417

theorem quadratic_expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y = 10 := by
  sorry

end quadratic_expression_evaluation_l2224_222417


namespace sum_product_nonpositive_l2224_222476

theorem sum_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_product_nonpositive_l2224_222476


namespace prob_king_ace_ten_l2224_222462

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of Tens in a standard deck -/
def NumTens : ℕ := 4

/-- Probability of drawing a King, then an Ace, then a 10 from a standard deck -/
theorem prob_king_ace_ten (deck : ℕ) (kings aces tens : ℕ) : 
  deck = StandardDeck → kings = NumKings → aces = NumAces → tens = NumTens →
  (kings : ℚ) / deck * aces / (deck - 1) * tens / (deck - 2) = 8 / 16575 := by
sorry

end prob_king_ace_ten_l2224_222462


namespace employed_females_percentage_l2224_222439

theorem employed_females_percentage (total_employed_percent : ℝ) (employed_males_percent : ℝ)
  (h1 : total_employed_percent = 64)
  (h2 : employed_males_percent = 48) :
  (total_employed_percent - employed_males_percent) / total_employed_percent * 100 = 25 := by
  sorry

end employed_females_percentage_l2224_222439


namespace a_range_l2224_222465

def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {1, 3, a}

theorem a_range (a : ℝ) : (A ∩ B a).Nonempty → a ∈ A := by
  sorry

end a_range_l2224_222465


namespace fourth_root_of_46656000_l2224_222429

theorem fourth_root_of_46656000 : (46656000 : ℝ) ^ (1/4 : ℝ) = 216 := by
  sorry

end fourth_root_of_46656000_l2224_222429


namespace inscribed_square_area_l2224_222446

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/8 = 1

-- Define a square inscribed in the ellipse
structure InscribedSquare where
  side : ℝ
  vertex_on_ellipse : ellipse (side/2) (side/2)

-- Theorem statement
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 32/3 := by
  sorry

end inscribed_square_area_l2224_222446


namespace inequality_proof_l2224_222412

theorem inequality_proof (x : ℝ) : 1 + 2 * x^2 ≥ 2 * x + x^2 := by
  sorry

end inequality_proof_l2224_222412


namespace average_age_problem_l2224_222415

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →
  b = 23 →
  (a + c) / 2 = 29 := by
sorry

end average_age_problem_l2224_222415


namespace solve_composite_function_equation_l2224_222464

theorem solve_composite_function_equation (a : ℝ) 
  (h : ℝ → ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_def : ∀ x, h x = x + 2)
  (f_def : ∀ x, f x = 2 * x + 3)
  (g_def : ∀ x, g x = x^2 - 5)
  (a_pos : a > 0)
  (eq : h (f (g a)) = 12) :
  a = Real.sqrt (17 / 2) := by
sorry

end solve_composite_function_equation_l2224_222464


namespace max_value_theorem_l2224_222460

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x^2 * y * Real.sqrt 6 + 8 * y^2 * z ≤ Real.sqrt (144/35) + Real.sqrt (88/35) := by
sorry

end max_value_theorem_l2224_222460


namespace quadratic_coefficient_l2224_222405

theorem quadratic_coefficient (c : ℝ) : (5 : ℝ)^2 + c * 5 + 45 = 0 → c = -14 := by
  sorry

end quadratic_coefficient_l2224_222405


namespace single_windows_upstairs_correct_number_of_single_windows_l2224_222496

theorem single_windows_upstairs 
  (double_windows : ℕ) 
  (panels_per_double : ℕ) 
  (panels_per_single : ℕ) 
  (total_panels : ℕ) : ℕ :=
  let downstairs_panels := double_windows * panels_per_double
  let upstairs_panels := total_panels - downstairs_panels
  upstairs_panels / panels_per_single

theorem correct_number_of_single_windows :
  single_windows_upstairs 6 4 4 80 = 14 := by
  sorry

end single_windows_upstairs_correct_number_of_single_windows_l2224_222496


namespace saramago_readers_l2224_222435

theorem saramago_readers (total_workers : ℕ) (kureishi_readers : ℚ) 
  (both_readers : ℕ) (s : ℚ) : 
  total_workers = 40 →
  kureishi_readers = 5/8 →
  both_readers = 2 →
  (s * total_workers - both_readers - 1 : ℚ) = 
    (total_workers * (1 - kureishi_readers - s) : ℚ) →
  s = 9/40 := by sorry

end saramago_readers_l2224_222435


namespace vector_magnitude_l2224_222421

theorem vector_magnitude (a b : ℝ × ℝ) : 
  (3 • a - 2 • b) • (5 • a + b) = 0 → 
  a • b = 1/7 → 
  ‖a‖ = 1 → 
  ‖b‖ = Real.sqrt 7 := by
  sorry

end vector_magnitude_l2224_222421


namespace intersection_of_S_and_T_l2224_222424

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | |x| < 5}
def T : Set ℝ := {x : ℝ | (x + 7) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by
  sorry

end intersection_of_S_and_T_l2224_222424


namespace max_annual_average_profit_l2224_222427

def profit_function (x : ℕ+) : ℚ := -x^2 + 18*x - 25

def annual_average_profit (x : ℕ+) : ℚ := (profit_function x) / x

theorem max_annual_average_profit :
  ∃ (x : ℕ+), (∀ (y : ℕ+), annual_average_profit y ≤ annual_average_profit x) ∧
              x = 5 ∧
              annual_average_profit x = 8 := by
  sorry

end max_annual_average_profit_l2224_222427


namespace quadratic_inequality_solution_l2224_222493

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 := by
  sorry

end quadratic_inequality_solution_l2224_222493


namespace function_properties_l2224_222433

noncomputable def m (a : ℝ) (t : ℝ) : ℝ := (1/2) * a * t^2 + t - a

noncomputable def g (a : ℝ) : ℝ :=
  if a > -1/2 then a + 2
  else if a > -Real.sqrt 2 / 2 then -a - 1 / (2 * a)
  else Real.sqrt 2

theorem function_properties (a : ℝ) :
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → 
    ∃ x : ℝ, m a t = a * Real.sqrt (1 - x^2) + Real.sqrt (1 + x) + Real.sqrt (1 - x)) ∧
  (∀ t : ℝ, Real.sqrt 2 ≤ t ∧ t ≤ 2 → m a t ≤ g a) ∧
  (a ≥ -Real.sqrt 2 → (g a = g (1/a) ↔ (-Real.sqrt 2 ≤ a ∧ a ≤ -Real.sqrt 2 / 2) ∨ a = 1)) :=
sorry

end function_properties_l2224_222433


namespace reading_materials_cost_l2224_222473

/-- The total cost of purchasing reading materials -/
def total_cost (a b : ℕ) : ℕ := 10 * a + 8 * b

/-- Theorem: The total cost of purchasing 'a' copies of type A reading materials
    at 10 yuan per copy and 'b' copies of type B reading materials at 8 yuan
    per copy is equal to 10a + 8b yuan. -/
theorem reading_materials_cost (a b : ℕ) :
  total_cost a b = 10 * a + 8 * b := by
  sorry

end reading_materials_cost_l2224_222473


namespace rachel_apples_remaining_l2224_222436

/-- The number of apples remaining on trees after picking -/
def apples_remaining (num_trees : ℕ) (apples_per_tree : ℕ) (initial_total : ℕ) : ℕ :=
  initial_total - (num_trees * apples_per_tree)

/-- Theorem: The number of apples remaining on Rachel's trees is 9 -/
theorem rachel_apples_remaining :
  apples_remaining 3 8 33 = 9 := by
  sorry

end rachel_apples_remaining_l2224_222436


namespace star_equation_has_two_distinct_real_roots_l2224_222489

-- Define the ☆ operation
def star (m n : ℝ) : ℝ := m * n^2 - m * n - 1

-- Theorem statement
theorem star_equation_has_two_distinct_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ star 1 x₁ = 0 ∧ star 1 x₂ = 0 :=
by sorry

end star_equation_has_two_distinct_real_roots_l2224_222489


namespace earlier_movie_savings_l2224_222470

/-- Calculates the savings when attending an earlier movie with discounts -/
def calculate_savings (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) 
  (ticket_discount_percent : ℝ) (food_discount_percent : ℝ) : ℝ :=
  (evening_ticket_cost * ticket_discount_percent) + 
  (food_combo_cost * food_discount_percent)

/-- Proves that the savings for the earlier movie is $7 -/
theorem earlier_movie_savings :
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount_percent : ℝ := 0.2
  let food_discount_percent : ℝ := 0.5
  calculate_savings evening_ticket_cost food_combo_cost 
    ticket_discount_percent food_discount_percent = 7 := by
  sorry

end earlier_movie_savings_l2224_222470


namespace train_crossing_time_l2224_222414

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 250 →
  platform_length = 200 →
  train_speed_kmph = 90 →
  (train_length + platform_length) / (train_speed_kmph * 1000 / 3600) = 18 := by
  sorry

end train_crossing_time_l2224_222414


namespace club_participation_theorem_l2224_222475

universe u

def club_participation (n : ℕ) : Prop :=
  ∃ (U A B C : Finset ℕ),
    Finset.card U = 40 ∧
    Finset.card A = 22 ∧
    Finset.card B = 16 ∧
    Finset.card C = 20 ∧
    Finset.card (A ∩ B) = 8 ∧
    Finset.card (B ∩ C) = 6 ∧
    Finset.card (A ∩ C) = 10 ∧
    Finset.card (A ∩ B ∩ C) = 2 ∧
    Finset.card (A \ (B ∪ C) ∪ B \ (A ∪ C) ∪ C \ (A ∪ B)) = 16 ∧
    Finset.card (U \ (A ∪ B ∪ C)) = 4

theorem club_participation_theorem : club_participation 40 := by
  sorry

end club_participation_theorem_l2224_222475


namespace xyz_value_l2224_222447

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  x * y * z = 4 := by
sorry

end xyz_value_l2224_222447


namespace sphere_quarter_sphere_radius_l2224_222463

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 4 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 * Real.rpow 2 (1/3) :=
sorry

end sphere_quarter_sphere_radius_l2224_222463


namespace problem_solution_l2224_222423

theorem problem_solution (x y : ℚ) : 
  x / y = 15 / 3 → y = 27 → x = 135 := by
  sorry

end problem_solution_l2224_222423


namespace line_through_circle_center_parallel_to_line_l2224_222485

/-- The equation of a line passing through the center of a circle and parallel to another line -/
theorem line_through_circle_center_parallel_to_line :
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + y^2 - 2*x + 2*y = 0
  let parallel_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y = 0
  let result_line_eq : ℝ → ℝ → Prop := λ x y => 2*x - y - 3 = 0
  ∃ (center_x center_y : ℝ),
    (∀ x y, circle_eq x y ↔ (x - center_x)^2 + (y - center_y)^2 = (center_x^2 + center_y^2)) →
    (∀ x y, result_line_eq x y ↔ y - center_y = 2 * (x - center_x)) →
    (∀ x₁ y₁ x₂ y₂, parallel_line_eq x₁ y₁ ∧ parallel_line_eq x₂ y₂ → y₂ - y₁ = 2 * (x₂ - x₁)) →
    result_line_eq center_x center_y :=
by
  sorry

end line_through_circle_center_parallel_to_line_l2224_222485


namespace min_moves_for_checkerboard_l2224_222404

/-- Represents a cell in the grid -/
inductive Cell
| White
| Black

/-- Represents a 6x6 grid -/
def Grid := Fin 6 → Fin 6 → Cell

/-- Represents a move (changing color of two adjacent cells) -/
structure Move where
  row : Fin 6
  col : Fin 6
  horizontal : Bool

/-- Defines a checkerboard pattern -/
def isCheckerboard (g : Grid) : Prop :=
  ∀ i j, g i j = if (i.val + j.val) % 2 = 0 then Cell.White else Cell.Black

/-- Applies a move to a grid -/
def applyMove (g : Grid) (m : Move) : Grid :=
  sorry

/-- Counts the number of black cells in a grid -/
def blackCellCount (g : Grid) : Nat :=
  sorry

theorem min_moves_for_checkerboard :
  ∀ (initial : Grid) (moves : List Move),
    (∀ i j, initial i j = Cell.White) →
    isCheckerboard (moves.foldl applyMove initial) →
    moves.length ≥ 18 :=
  sorry

end min_moves_for_checkerboard_l2224_222404


namespace symmetric_points_on_circumcircle_l2224_222406

-- Define a triangle
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a point symmetric to another point with respect to a line
def symmetric_point (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the circumcircle of a triangle
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem symmetric_points_on_circumcircle (t : Triangle) :
  let H := orthocenter t
  let A1 := symmetric_point H (t.B, t.C)
  let B1 := symmetric_point H (t.C, t.A)
  let C1 := symmetric_point H (t.A, t.B)
  A1 ∈ circumcircle t ∧ B1 ∈ circumcircle t ∧ C1 ∈ circumcircle t := by
  sorry

end symmetric_points_on_circumcircle_l2224_222406


namespace find_p_l2224_222407

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (p w : ℂ) : Prop := 10 * p - w = 50000

-- State the theorem
theorem find_p :
  ∀ (p w : ℂ),
  equation p w →
  (10 : ℂ) = 2 →
  w = 10 + 250 * i →
  p = 5001 + 25 * i :=
by sorry

end find_p_l2224_222407


namespace rational_square_plus_one_positive_l2224_222432

theorem rational_square_plus_one_positive (x : ℚ) : x^2 + 1 > 0 := by
  sorry

end rational_square_plus_one_positive_l2224_222432


namespace max_value_sum_of_square_roots_l2224_222477

theorem max_value_sum_of_square_roots (x : ℝ) (h : x ∈ Set.Icc (-36) 36) :
  Real.sqrt (36 + x) + Real.sqrt (36 - x) ≤ 12 ∧
  ∃ y ∈ Set.Icc (-36) 36, Real.sqrt (36 + y) + Real.sqrt (36 - y) = 12 :=
by sorry

end max_value_sum_of_square_roots_l2224_222477


namespace hyperbola_asymptote_angle_l2224_222445

/-- The angle between asymptotes of a hyperbola -/
theorem hyperbola_asymptote_angle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  hyperbola x y ∧ e = Real.sqrt (1 + b^2 / a^2) →
  ∃ θ : ℝ, θ = π / 3 ∧ θ = 2 * Real.arctan (b / a) :=
by sorry

end hyperbola_asymptote_angle_l2224_222445


namespace quadratic_max_value_l2224_222450

/-- The quadratic function f(x) = -x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 - 2*x + 3

theorem quadratic_max_value (a : ℝ) :
  (∀ x ∈ Set.Icc a 2, f x ≤ 15/4) ∧ 
  (∃ x ∈ Set.Icc a 2, f x = 15/4) →
  a = -1/2 :=
sorry

end quadratic_max_value_l2224_222450


namespace subcommittees_count_l2224_222471

def planning_committee_size : ℕ := 10
def teacher_count : ℕ := 4
def subcommittee_size : ℕ := 4

/-- The number of distinct subcommittees with at least one teacher -/
def subcommittees_with_teacher : ℕ :=
  Nat.choose planning_committee_size subcommittee_size -
  Nat.choose (planning_committee_size - teacher_count) subcommittee_size

theorem subcommittees_count :
  subcommittees_with_teacher = 195 :=
sorry

end subcommittees_count_l2224_222471


namespace evaluate_expression_l2224_222456

theorem evaluate_expression : 3 - 6 * (7 - 2^3)^2 = -3 := by
  sorry

end evaluate_expression_l2224_222456


namespace abs_equal_necessary_not_sufficient_l2224_222425

theorem abs_equal_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end abs_equal_necessary_not_sufficient_l2224_222425


namespace myPolygonArea_l2224_222482

/-- A point in a 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A polygon defined by a list of points -/
def Polygon := List Point

/-- The polygon in question -/
def myPolygon : Polygon := [
  {x := 0, y := 0},
  {x := 0, y := 30},
  {x := 30, y := 30},
  {x := 30, y := 0}
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℤ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area of myPolygon is 15 square units -/
theorem myPolygonArea : calculateArea myPolygon = 15 := by
  sorry

end myPolygonArea_l2224_222482


namespace domain_of_g_l2224_222434

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the original domain of f
def original_domain : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 :=
sorry

end domain_of_g_l2224_222434


namespace product_of_four_is_perfect_square_l2224_222472

theorem product_of_four_is_perfect_square 
  (nums : Finset ℕ) 
  (h_card : nums.card = 48) 
  (h_primes : (nums.prod id).factorization.support.card = 10) : 
  ∃ (subset : Finset ℕ), subset ⊆ nums ∧ subset.card = 4 ∧ 
  ∃ (m : ℕ), (subset.prod id) = m^2 := by
  sorry

end product_of_four_is_perfect_square_l2224_222472


namespace product_increase_l2224_222486

theorem product_increase (a b : ℕ) (h1 : a * b = 72) : a * (10 * b) = 720 := by
  sorry

end product_increase_l2224_222486


namespace box_properties_l2224_222418

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  white : ℕ
  red : ℕ
  yellow : ℕ

/-- The given ball counts in the box -/
def box : BallCounts := { white := 1, red := 2, yellow := 3 }

/-- The total number of balls in the box -/
def totalBalls (b : BallCounts) : ℕ := b.white + b.red + b.yellow

/-- The number of possible outcomes when drawing 1 ball -/
def possibleOutcomes (b : BallCounts) : ℕ := 
  (if b.white > 0 then 1 else 0) + 
  (if b.red > 0 then 1 else 0) + 
  (if b.yellow > 0 then 1 else 0)

/-- The probability of drawing a ball of a specific color -/
def probability (b : BallCounts) (color : ℕ) : ℚ :=
  color / (totalBalls b : ℚ)

theorem box_properties : 
  (possibleOutcomes box = 3) ∧ 
  (probability box box.yellow > probability box box.red ∧ 
   probability box box.yellow > probability box box.white) ∧
  (probability box box.white + probability box box.yellow = 2/3) := by
  sorry

end box_properties_l2224_222418


namespace range_of_a_l2224_222479

def p (x : ℝ) : Prop := (4 * x - 3)^2 ≤ 1

def q (a x : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ a : ℝ, (∀ x : ℝ, p x → q a x) ∧ 
  (∃ x : ℝ, ¬p x ∧ q a x)) → 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2) ∧ 
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1/2 → 
    (∀ x : ℝ, p x → q a x) ∧ 
    (∃ x : ℝ, ¬p x ∧ q a x)) :=
by sorry

end range_of_a_l2224_222479


namespace tangent_line_at_one_max_value_min_value_l2224_222469

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 6*x + 5

-- Define the interval
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Theorem for the tangent line at x = 1
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 3*x - y - 3 = 0 :=
sorry

-- Theorem for the maximum value
theorem max_value :
  ∃ x ∈ interval, f x = 5 + 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≤ f x :=
sorry

-- Theorem for the minimum value
theorem min_value :
  ∃ x ∈ interval, f x = 5 - 4 * Real.sqrt 2 ∧ ∀ y ∈ interval, f y ≥ f x :=
sorry

end tangent_line_at_one_max_value_min_value_l2224_222469


namespace first_repeat_l2224_222409

/-- The number of points on the circle -/
def n : ℕ := 2021

/-- The function that calculates the position of the nth marked point -/
def f (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The theorem stating that 66 is the smallest positive integer b such that
    there exists an a < b where f(a) ≡ f(b) (mod n) -/
theorem first_repeat : 
  ∀ b < 66, ¬∃ a < b, f a % n = f b % n ∧ 
  ∃ a < 66, f a % n = f 66 % n :=
sorry

end first_repeat_l2224_222409


namespace email_sending_combinations_l2224_222494

theorem email_sending_combinations (num_addresses : ℕ) (num_emails : ℕ) : 
  num_addresses = 3 → num_emails = 5 → num_addresses ^ num_emails = 243 :=
by sorry

end email_sending_combinations_l2224_222494


namespace meal_combinations_l2224_222461

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of dishes Camille avoids -/
def avoided_dishes : ℕ := 2

/-- The number of dishes Camille can choose from -/
def camille_choices : ℕ := menu_items - avoided_dishes

theorem meal_combinations : menu_items * camille_choices = 195 := by
  sorry

end meal_combinations_l2224_222461


namespace number_of_boys_l2224_222481

theorem number_of_boys (total : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 100 →
  boys + girls = total →
  girls = boys * total / 100 →
  boys = 50 := by
sorry

end number_of_boys_l2224_222481


namespace crayon_purchase_worth_l2224_222497

/-- Calculates the total worth of crayons after a discounted purchase -/
theorem crayon_purchase_worth
  (initial_packs : ℕ)
  (additional_packs : ℕ)
  (regular_price : ℝ)
  (discount_percent : ℝ)
  (h1 : initial_packs = 4)
  (h2 : additional_packs = 2)
  (h3 : regular_price = 2.5)
  (h4 : discount_percent = 15)
  : ℝ := by
  sorry

#check crayon_purchase_worth

end crayon_purchase_worth_l2224_222497


namespace circle_existence_condition_l2224_222487

theorem circle_existence_condition (x y c : ℝ) : 
  (∃ h k r, (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔ 
  (x^2 + y^2 + 4*x - 2*y - 5*c = 0 → c > -1) :=
sorry

end circle_existence_condition_l2224_222487


namespace max_value_equality_l2224_222499

theorem max_value_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end max_value_equality_l2224_222499


namespace sandro_has_three_sons_l2224_222492

/-- Represents the number of sons Sandro has -/
def num_sons : ℕ := 3

/-- Represents the number of daughters Sandro has -/
def num_daughters : ℕ := 6 * num_sons

/-- The total number of children Sandro has -/
def total_children : ℕ := 21

/-- Theorem stating that Sandro has 3 sons, given the conditions -/
theorem sandro_has_three_sons : 
  (num_daughters = 6 * num_sons) ∧ 
  (num_sons + num_daughters = total_children) → 
  num_sons = 3 := by
  sorry

end sandro_has_three_sons_l2224_222492


namespace system_of_inequalities_l2224_222408

theorem system_of_inequalities (x : ℝ) :
  (2 * x + 1 < 3) ∧ (x / 2 + (1 - 3 * x) / 4 ≤ 1) → -3 ≤ x ∧ x < 1 := by
  sorry

end system_of_inequalities_l2224_222408


namespace stating_fencers_count_correct_l2224_222437

/-- The number of fencers participating in the championship. -/
def num_fencers : ℕ := 9

/-- The number of possibilities for awarding first and second place medals. -/
def num_possibilities : ℕ := 72

/-- 
Theorem stating that the number of fencers is correct given the number of possibilities 
for awarding first and second place medals.
-/
theorem fencers_count_correct : 
  num_fencers * (num_fencers - 1) = num_possibilities := by
  sorry

#check fencers_count_correct

end stating_fencers_count_correct_l2224_222437


namespace system_of_inequalities_solution_l2224_222400

theorem system_of_inequalities_solution (x : ℝ) : 
  ((x - 1) / 2 < 2 * x + 1 ∧ -3 * (1 - x) ≥ -4) ↔ x ≥ -1/3 := by
  sorry

end system_of_inequalities_solution_l2224_222400


namespace no_solution_for_specific_k_l2224_222478

theorem no_solution_for_specific_k (p : ℕ) (hp : Prime p) (hp_mod : p % 4 = 3) :
  ¬ ∃ (n m : ℕ+), (n.val^2 + m.val^2 : ℚ) / (m.val^4 + n.val) = p^2 := by
  sorry

end no_solution_for_specific_k_l2224_222478


namespace friday_temperature_l2224_222431

theorem friday_temperature
  (temp : Fin 5 → ℝ)
  (avg_mon_to_thu : (temp 0 + temp 1 + temp 2 + temp 3) / 4 = 48)
  (avg_tue_to_fri : (temp 1 + temp 2 + temp 3 + temp 4) / 4 = 46)
  (monday_temp : temp 0 = 42) :
  temp 4 = 34 :=
by
  sorry

end friday_temperature_l2224_222431


namespace courtyard_paving_l2224_222410

/-- Represents the dimensions of a rectangular area in centimeters -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts meters to centimeters -/
def meters_to_cm (m : ℕ) : ℕ := m * 100

/-- The dimensions of the courtyard in meters -/
def courtyard_m : Dimensions := ⟨30, 16⟩

/-- The dimensions of the courtyard in centimeters -/
def courtyard_cm : Dimensions := ⟨meters_to_cm courtyard_m.length, meters_to_cm courtyard_m.width⟩

/-- The dimensions of a single brick in centimeters -/
def brick : Dimensions := ⟨20, 10⟩

/-- Calculates the number of bricks needed to cover an area -/
def bricks_needed (area_to_cover : ℕ) (brick_size : ℕ) : ℕ := area_to_cover / brick_size

theorem courtyard_paving :
  bricks_needed (area courtyard_cm) (area brick) = 24000 := by
  sorry

end courtyard_paving_l2224_222410


namespace corners_removed_cube_edges_l2224_222498

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the solid formed by removing smaller cubes from corners of a larger cube -/
structure CornersRemovedCube where
  originalCube : Cube
  removedCubeSideLength : ℝ
  removedCubeSideLength_pos : removedCubeSideLength > 0
  validRemoval : removedCubeSideLength < originalCube.sideLength / 2

/-- Calculates the number of edges in the resulting solid after removing smaller cubes from corners -/
def edgesAfterRemoval (c : CornersRemovedCube) : ℕ :=
  sorry

/-- Theorem stating that removing cubes of side length 2 from corners of a cube with side length 6 results in a solid with 36 edges -/
theorem corners_removed_cube_edges :
  let originalCube : Cube := ⟨6, by norm_num⟩
  let cornersRemovedCube : CornersRemovedCube := ⟨originalCube, 2, by norm_num, by norm_num⟩
  edgesAfterRemoval cornersRemovedCube = 36 :=
sorry

end corners_removed_cube_edges_l2224_222498


namespace fraction_equals_zero_l2224_222455

theorem fraction_equals_zero (x : ℝ) : 
  (|x| - 2) / (x - 2) = 0 ∧ x - 2 ≠ 0 ↔ x = -2 := by
sorry

end fraction_equals_zero_l2224_222455


namespace batsman_average_increase_l2224_222483

/-- Represents a batsman's performance over a series of innings -/
structure BatsmanPerformance where
  innings : ℕ
  totalRuns : ℕ
  neverNotOut : Bool

/-- Calculates the average runs per innings -/
def average (perf : BatsmanPerformance) : ℚ :=
  perf.totalRuns / perf.innings

/-- Represents the change in a batsman's performance after an additional innings -/
structure PerformanceChange where
  before : BatsmanPerformance
  runsScored : ℕ
  newAverage : ℚ

/-- Calculates the increase in average after an additional innings -/
def averageIncrease (change : PerformanceChange) : ℚ :=
  change.newAverage - average change.before

theorem batsman_average_increase :
  ∀ (perf : BatsmanPerformance) (change : PerformanceChange),
    perf.innings = 11 →
    perf.neverNotOut = true →
    change.before = perf →
    change.runsScored = 60 →
    change.newAverage = 38 →
    averageIncrease change = 2 := by
  sorry

end batsman_average_increase_l2224_222483


namespace slope_intercept_sum_l2224_222438

/-- Given points A, B, C, and D in a Cartesian plane, where D is the midpoint of AB,
    prove that the sum of the slope and y-intercept of the line passing through C and D is 3.6 -/
theorem slope_intercept_sum (A B C D : ℝ × ℝ) : 
  A = (0, 8) →
  B = (0, 0) →
  C = (10, 0) →
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  let slope := (D.2 - C.2) / (D.1 - C.1)
  let y_intercept := D.2
  slope + y_intercept = 3.6 := by
sorry


end slope_intercept_sum_l2224_222438


namespace arithmetic_sequence_general_term_l2224_222451

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d = 3)
  (h3 : a 4 = 14) :
  ∀ n, a n = 3 * n + 2 :=
sorry

end arithmetic_sequence_general_term_l2224_222451


namespace rahul_share_l2224_222422

/-- Calculates the share of a worker given the total payment and the time taken by both workers --/
def calculateShare (totalPayment : ℚ) (time1 : ℚ) (time2 : ℚ) : ℚ :=
  let combinedRate := 1 / time1 + 1 / time2
  let share := (1 / time1) / combinedRate
  share * totalPayment

/-- Proves that Rahul's share is $60 given the problem conditions --/
theorem rahul_share :
  let rahulTime : ℚ := 3
  let rajeshTime : ℚ := 2
  let totalPayment : ℚ := 150
  calculateShare totalPayment rahulTime rajeshTime = 60 := by
sorry

#eval calculateShare 150 3 2

end rahul_share_l2224_222422


namespace remaining_money_correct_l2224_222458

structure Currency where
  usd : ℚ
  eur : ℚ
  gbp : ℚ

def initial_amount : Currency := ⟨5.10, 8.75, 10.30⟩

def spend_usd (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd - amount, c.eur, c.gbp⟩

def spend_eur (amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur - amount, c.gbp⟩

def exchange_gbp_to_eur (gbp_amount : ℚ) (eur_amount : ℚ) (c : Currency) : Currency :=
  ⟨c.usd, c.eur + eur_amount, c.gbp - gbp_amount⟩

def final_amount : Currency :=
  initial_amount
  |> spend_usd 1.05
  |> spend_usd 2.00
  |> spend_eur 3.25
  |> exchange_gbp_to_eur 5.00 5.60

theorem remaining_money_correct :
  final_amount.usd = 2.05 ∧
  final_amount.eur = 11.10 ∧
  final_amount.gbp = 5.30 := by
  sorry


end remaining_money_correct_l2224_222458


namespace converse_x_squared_greater_than_one_l2224_222449

theorem converse_x_squared_greater_than_one (x : ℝ) :
  x^2 > 1 → (x < -1 ∨ x > 1) :=
sorry

end converse_x_squared_greater_than_one_l2224_222449


namespace solution_equation_l2224_222419

theorem solution_equation (x : ℝ) (k : ℤ) : 
  (8.492 * (Real.log (Real.sin x) / Real.log (Real.sin x * Real.cos x)) * 
           (Real.log (Real.cos x) / Real.log (Real.sin x * Real.cos x)) = 1/4) →
  (Real.sin x > 0) →
  (x = π/4 * (8 * ↑k + 1)) :=
by sorry

end solution_equation_l2224_222419


namespace polynomial_division_theorem_l2224_222468

/-- A polynomial is monic if its leading coefficient is 1. -/
def Monic (p : Polynomial ℤ) : Prop :=
  p.leadingCoeff = 1

/-- A polynomial is non-constant if its degree is greater than 0. -/
def NonConstant (p : Polynomial ℤ) : Prop :=
  p.degree > 0

/-- P(n) divides Q(n) in ℤ -/
def DividesAtInteger (P Q : Polynomial ℤ) (n : ℤ) : Prop :=
  ∃ k : ℤ, Q.eval n = k * P.eval n

/-- There are infinitely many integers n such that P(n) divides Q(n) in ℤ -/
def InfinitelyManyDivisions (P Q : Polynomial ℤ) : Prop :=
  ∀ m : ℕ, ∃ n : ℤ, n > m ∧ DividesAtInteger P Q n

theorem polynomial_division_theorem (P Q : Polynomial ℤ) 
  (h_monic_P : Monic P) (h_monic_Q : Monic Q)
  (h_non_const_P : NonConstant P) (h_non_const_Q : NonConstant Q)
  (h_infinite_divisions : InfinitelyManyDivisions P Q) :
  P ∣ Q :=
sorry

end polynomial_division_theorem_l2224_222468


namespace paint_mixing_l2224_222430

/-- Represents the mixing of two paints to achieve a target yellow percentage -/
theorem paint_mixing (light_green_volume : ℝ) (light_green_yellow_percent : ℝ)
  (dark_green_yellow_percent : ℝ) (target_yellow_percent : ℝ) :
  light_green_volume = 5 →
  light_green_yellow_percent = 0.2 →
  dark_green_yellow_percent = 0.4 →
  target_yellow_percent = 0.25 →
  ∃ dark_green_volume : ℝ,
    dark_green_volume = 5 / 3 ∧
    (light_green_volume * light_green_yellow_percent + dark_green_volume * dark_green_yellow_percent) /
      (light_green_volume + dark_green_volume) = target_yellow_percent :=
by sorry

end paint_mixing_l2224_222430


namespace smallest_n_for_candy_purchase_l2224_222474

theorem smallest_n_for_candy_purchase : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 → (20 * m) % 12 = 0 ∧ (20 * m) % 14 = 0 ∧ (20 * m) % 15 = 0 → m ≥ n) ∧
  (20 * n) % 12 = 0 ∧ (20 * n) % 14 = 0 ∧ (20 * n) % 15 = 0 ∧ n = 21 := by
  sorry

end smallest_n_for_candy_purchase_l2224_222474


namespace nathaniel_win_probability_l2224_222441

-- Define the game state
structure GameState where
  tally : ℕ
  currentPlayer : Bool  -- True for Nathaniel, False for Obediah

-- Define the probability of winning for a given game state
def winProbability (state : GameState) : ℚ :=
  sorry

-- Define the theorem
theorem nathaniel_win_probability :
  winProbability ⟨0, true⟩ = 5/11 := by sorry

end nathaniel_win_probability_l2224_222441


namespace slow_car_speed_is_correct_l2224_222428

/-- The speed of the slow car in km/h -/
def slow_car_speed : ℝ := 40

/-- The speed of the fast car in km/h -/
def fast_car_speed : ℝ := 1.5 * slow_car_speed

/-- The distance to the memorial hall in km -/
def distance : ℝ := 60

/-- The time difference between departures in hours -/
def time_difference : ℝ := 0.5

theorem slow_car_speed_is_correct :
  (distance / slow_car_speed) - (distance / fast_car_speed) = time_difference :=
sorry

end slow_car_speed_is_correct_l2224_222428
