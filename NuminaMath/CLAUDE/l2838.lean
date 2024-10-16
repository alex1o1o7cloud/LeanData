import Mathlib

namespace NUMINAMATH_CALUDE_thirteenth_number_with_digit_sum_12_l2838_283878

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 12 -/
def nth_number_with_digit_sum_12 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 13th number with digit sum 12 is 174 -/
theorem thirteenth_number_with_digit_sum_12 : 
  nth_number_with_digit_sum_12 13 = 174 := by sorry

end NUMINAMATH_CALUDE_thirteenth_number_with_digit_sum_12_l2838_283878


namespace NUMINAMATH_CALUDE_corrected_mean_l2838_283834

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 60 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / n = 36.74 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l2838_283834


namespace NUMINAMATH_CALUDE_point_movement_to_origin_l2838_283859

theorem point_movement_to_origin (a b : ℝ) :
  (2 * a - 2 = 0) ∧ (-3 * b - 3 = 0) →
  (2 * a = 2) ∧ (-3 * b = 3) :=
by sorry

end NUMINAMATH_CALUDE_point_movement_to_origin_l2838_283859


namespace NUMINAMATH_CALUDE_function_inequality_l2838_283892

theorem function_inequality (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_cond : ∀ x, (x^2 - 3*x + 2) * deriv f x ≤ 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2838_283892


namespace NUMINAMATH_CALUDE_xyz_value_l2838_283840

theorem xyz_value (a b c x y z : ℂ) 
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + z * x = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2838_283840


namespace NUMINAMATH_CALUDE_six_digit_square_number_puzzle_l2838_283874

theorem six_digit_square_number_puzzle :
  ∃ (n x y : ℕ), 
    100000 ≤ n^2 ∧ n^2 < 1000000 ∧
    10 ≤ x ∧ x ≤ 99 ∧
    0 ≤ y ∧ y ≤ 9 ∧
    n^2 = 10101 * x + y^2 ∧
    (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_square_number_puzzle_l2838_283874


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2838_283838

theorem min_value_of_expression :
  ∃ (min : ℝ), min = -6452.25 ∧
  ∀ (x : ℝ), (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2838_283838


namespace NUMINAMATH_CALUDE_inverse_fraction_ratio_l2838_283860

noncomputable def g (x : ℝ) : ℝ := (3 * x - 2) / (x + 4)

theorem inverse_fraction_ratio (a b c d : ℝ) :
  (∀ x, g (((a * x + b) / (c * x + d)) : ℝ) = x) →
  a / c = -4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_fraction_ratio_l2838_283860


namespace NUMINAMATH_CALUDE_elimination_tournament_sequences_l2838_283849

def team_size : ℕ := 7

/-- The number of possible sequences in the elimination tournament -/
def elimination_sequences (n : ℕ) : ℕ :=
  2 * (Nat.choose (2 * n - 1) (n - 1))

/-- The theorem stating the number of possible sequences for the given problem -/
theorem elimination_tournament_sequences :
  elimination_sequences team_size = 3432 := by
  sorry

end NUMINAMATH_CALUDE_elimination_tournament_sequences_l2838_283849


namespace NUMINAMATH_CALUDE_unknown_table_has_one_leg_l2838_283811

/-- The number of legs on the table with the unknown number of legs -/
def unknown_table_legs : ℕ := sorry

/-- The total number of legs in the room -/
def total_legs : ℕ := 40

/-- The number of legs on all furniture except the unknown table -/
def known_legs : ℕ := 
  4 * 4 +  -- 4 tables with 4 legs each
  1 * 4 +  -- 1 sofa with 4 legs
  2 * 4 +  -- 2 chairs with 4 legs each
  3 * 3 +  -- 3 tables with 3 legs each
  1 * 2    -- 1 rocking chair with 2 legs

theorem unknown_table_has_one_leg : 
  unknown_table_legs = 1 :=
by
  sorry

#check unknown_table_has_one_leg

end NUMINAMATH_CALUDE_unknown_table_has_one_leg_l2838_283811


namespace NUMINAMATH_CALUDE_cube_side_length_when_volume_equals_surface_area_l2838_283897

/-- For a cube where the numerical value of its volume equals the numerical value of its surface area, the side length is 6 units. -/
theorem cube_side_length_when_volume_equals_surface_area :
  ∀ s : ℝ, s > 0 → s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_when_volume_equals_surface_area_l2838_283897


namespace NUMINAMATH_CALUDE_labor_union_tree_planting_equation_l2838_283869

/-- Represents a labor union planting trees -/
structure LaborUnion where
  members : ℕ
  trees : ℕ

/-- The equation holds for a labor union's tree planting scenario -/
theorem labor_union_tree_planting_equation (union : LaborUnion) :
  (2 * union.members + 21 = union.trees) →
  (3 * union.members = union.trees + 24) →
  (2 * union.members + 21 = 3 * union.members - 24) :=
by
  sorry

end NUMINAMATH_CALUDE_labor_union_tree_planting_equation_l2838_283869


namespace NUMINAMATH_CALUDE_haley_recycling_cans_l2838_283861

theorem haley_recycling_cans (total_cans : ℕ) (difference : ℕ) (cans_in_bag : ℕ) :
  total_cans = 9 →
  difference = 2 →
  total_cans - cans_in_bag = difference →
  cans_in_bag = 7 := by
sorry

end NUMINAMATH_CALUDE_haley_recycling_cans_l2838_283861


namespace NUMINAMATH_CALUDE_staples_left_l2838_283898

def initial_staples : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

theorem staples_left : initial_staples - reports_stapled = 14 := by
  sorry

end NUMINAMATH_CALUDE_staples_left_l2838_283898


namespace NUMINAMATH_CALUDE_new_person_weight_l2838_283839

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 105 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2838_283839


namespace NUMINAMATH_CALUDE_even_power_iff_even_l2838_283825

theorem even_power_iff_even (n : ℕ) (hn : n ≠ 0) :
  Even (n^n) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_even_power_iff_even_l2838_283825


namespace NUMINAMATH_CALUDE_antonov_candy_count_l2838_283863

/-- The number of candies in a pack -/
def candies_per_pack : ℕ := 20

/-- The number of packs Antonov has left -/
def packs_left : ℕ := 2

/-- The number of packs Antonov gave away -/
def packs_given : ℕ := 1

/-- The total number of candies Antonov bought initially -/
def total_candies : ℕ := (packs_left + packs_given) * candies_per_pack

theorem antonov_candy_count : total_candies = 60 := by sorry

end NUMINAMATH_CALUDE_antonov_candy_count_l2838_283863


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2838_283862

/-- Given an angle α in the second quadrant, if the slope of the line 2x + (tan α)y + 1 = 0 is 8/3, 
    then cos α = -4/5 -/
theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 : Real) / Real.tan α = 8/3) →  -- slope of the line
  Real.cos α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2838_283862


namespace NUMINAMATH_CALUDE_total_missed_pitches_l2838_283847

/-- The number of pitches per token -/
def pitches_per_token : ℕ := 15

/-- The number of tokens Macy used -/
def macy_tokens : ℕ := 11

/-- The number of tokens Piper used -/
def piper_tokens : ℕ := 17

/-- The number of times Macy hit the ball -/
def macy_hits : ℕ := 50

/-- The number of times Piper hit the ball -/
def piper_hits : ℕ := 55

/-- Theorem stating the total number of missed pitches -/
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token - macy_hits) +
  (piper_tokens * pitches_per_token - piper_hits) = 315 := by
  sorry

end NUMINAMATH_CALUDE_total_missed_pitches_l2838_283847


namespace NUMINAMATH_CALUDE_equation_solutions_equation_solutions_unique_l2838_283801

theorem equation_solutions :
  (∃ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3)) ∧
  (∃ x : ℝ, x^2 - 4*x + 2 = 0) :=
by
  constructor
  · use -3/2
    sorry
  · use 2 + Real.sqrt 2
    sorry

theorem equation_solutions_unique :
  (∀ x : ℝ, (2*x + 3)^2 = 4*(2*x + 3) ↔ (x = -3/2 ∨ x = 1/2)) ∧
  (∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x = 2 + Real.sqrt 2 ∨ x = 2 - Real.sqrt 2)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end NUMINAMATH_CALUDE_equation_solutions_equation_solutions_unique_l2838_283801


namespace NUMINAMATH_CALUDE_used_books_count_l2838_283812

def total_books : ℕ := 30
def new_books : ℕ := 15

theorem used_books_count : total_books - new_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_used_books_count_l2838_283812


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2838_283821

theorem arithmetic_sequence_sum : ∀ (a₁ a_last d n : ℕ),
  a₁ = 1 →
  a_last = 23 →
  d = 2 →
  n = (a_last - a₁) / d + 1 →
  (n : ℝ) * (a₁ + a_last) / 2 = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2838_283821


namespace NUMINAMATH_CALUDE_quarters_to_dollars_l2838_283851

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total number of quarters -/
def total_quarters : ℕ := 8

/-- The dollar amount equivalent to the total number of quarters -/
def dollar_amount : ℚ := total_quarters / quarters_per_dollar

theorem quarters_to_dollars : dollar_amount = 2 := by
  sorry

end NUMINAMATH_CALUDE_quarters_to_dollars_l2838_283851


namespace NUMINAMATH_CALUDE_cubic_difference_division_l2838_283867

theorem cubic_difference_division (a b : ℝ) (ha : a = 6) (hb : b = 3) :
  (a^3 - b^3) / (a^2 + a*b + b^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_division_l2838_283867


namespace NUMINAMATH_CALUDE_square_of_binomial_l2838_283853

theorem square_of_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 4*x^2 + 12*x + a = (2*x + b)^2) → a = 9 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_l2838_283853


namespace NUMINAMATH_CALUDE_square_corners_l2838_283816

theorem square_corners (S : ℤ) : ∃ (A B C D : ℤ),
  A + B + 9 = S ∧
  B + C + 6 = S ∧
  D + C + 12 = S ∧
  D + A + 15 = S ∧
  A + C + 17 = S ∧
  A + B + C + D = 123 ∧
  A = 26 ∧ B = 37 ∧ C = 29 ∧ D = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_corners_l2838_283816


namespace NUMINAMATH_CALUDE_negation_of_p_l2838_283844

-- Define the proposition p
def p : Prop := ∀ a : ℝ, a ≥ 0 → ∃ x : ℝ, x^2 + a*x + 1 = 0

-- State the theorem
theorem negation_of_p : 
  ¬p ↔ ∃ a : ℝ, a ≥ 0 ∧ ¬∃ x : ℝ, x^2 + a*x + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_of_p_l2838_283844


namespace NUMINAMATH_CALUDE_sum_reciprocal_inequality_max_nonnegative_l2838_283837

-- Problem 1
theorem sum_reciprocal_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 9 := by
  sorry

-- Problem 2
theorem max_nonnegative (x : ℝ) :
  let a := x^2 - 1
  let b := 2*x + 2
  max a b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_inequality_max_nonnegative_l2838_283837


namespace NUMINAMATH_CALUDE_number_relationship_l2838_283808

theorem number_relationship (A B C : ℝ) 
  (h1 : B = 10)
  (h2 : A * B = 85)
  (h3 : B * C = 115)
  (h4 : B - A = C - B) :
  B - A = 1.5 ∧ C - B = 1.5 := by
sorry

end NUMINAMATH_CALUDE_number_relationship_l2838_283808


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2838_283842

theorem modulus_of_complex_fraction : Complex.abs ((2 - Complex.I) / (1 + Complex.I)) = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l2838_283842


namespace NUMINAMATH_CALUDE_ufo_convention_attendees_l2838_283885

theorem ufo_convention_attendees (total : ℕ) (male : ℕ) 
  (h1 : total = 120) 
  (h2 : male = 62) 
  (h3 : male > total - male) : 
  male - (total - male) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ufo_convention_attendees_l2838_283885


namespace NUMINAMATH_CALUDE_storks_joined_l2838_283845

theorem storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → final_total = 13 →
  final_total - (initial_birds + initial_storks) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_storks_joined_l2838_283845


namespace NUMINAMATH_CALUDE_fifth_term_equals_fourth_l2838_283875

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ+  -- first term
  r : ℕ+  -- common ratio

/-- The nth term of a geometric sequence -/
def nthTerm (seq : GeometricSequence) (n : ℕ) : ℕ+ :=
  seq.a * (seq.r ^ (n - 1))

theorem fifth_term_equals_fourth (seq : GeometricSequence) 
  (h1 : seq.a = 4)
  (h2 : nthTerm seq 4 = 324) :
  nthTerm seq 5 = 324 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_equals_fourth_l2838_283875


namespace NUMINAMATH_CALUDE_abc_inequality_l2838_283893

theorem abc_inequality (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2838_283893


namespace NUMINAMATH_CALUDE_odd_power_sum_is_prime_power_l2838_283884

theorem odd_power_sum_is_prime_power (n p x y k : ℕ) :
  Odd n →
  n > 1 →
  Prime p →
  Odd p →
  x^n + y^n = p^k →
  ∃ m : ℕ, n = p^m :=
by sorry

end NUMINAMATH_CALUDE_odd_power_sum_is_prime_power_l2838_283884


namespace NUMINAMATH_CALUDE_equipment_production_l2838_283830

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_A : ℕ) (products_B : ℕ) : 
  total = 4800 → 
  sample_size = 80 → 
  sample_A = 50 → 
  products_B = total - (total * sample_A / sample_size) →
  products_B = 1800 := by
sorry

end NUMINAMATH_CALUDE_equipment_production_l2838_283830


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2838_283809

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ
  side : ℝ

/-- The theorem stating the side length of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_side_length 
  (t : IsoscelesTrapezoid) 
  (h_area : t.area = 44)
  (h_base1 : t.base1 = 8)
  (h_base2 : t.base2 = 14) :
  t.side = 5 := by
  sorry

#check isosceles_trapezoid_side_length

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l2838_283809


namespace NUMINAMATH_CALUDE_total_money_earned_l2838_283850

def clementine_cookies : ℕ := 72
def jake_cookies : ℕ := 2 * clementine_cookies
def combined_cookies : ℕ := jake_cookies + clementine_cookies
def tory_cookies : ℕ := combined_cookies / 2
def total_cookies : ℕ := clementine_cookies + jake_cookies + tory_cookies
def price_per_cookie : ℕ := 2

theorem total_money_earned : total_cookies * price_per_cookie = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_money_earned_l2838_283850


namespace NUMINAMATH_CALUDE_rectangle_area_formula_l2838_283872

/-- Represents a rectangle with a specific ratio of length to width and diagonal length. -/
structure Rectangle where
  ratio_length : ℝ
  ratio_width : ℝ
  diagonal : ℝ
  ratio_condition : ratio_length / ratio_width = 5 / 2

/-- The theorem stating that the area of the rectangle can be expressed as (10/29)d^2 -/
theorem rectangle_area_formula (rect : Rectangle) : 
  ∃ (length width : ℝ), 
    length / width = rect.ratio_length / rect.ratio_width ∧
    length * width = (10/29) * rect.diagonal^2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_formula_l2838_283872


namespace NUMINAMATH_CALUDE_logarithm_inequality_l2838_283835

theorem logarithm_inequality (a b c : ℝ) (ha : a ≥ 2) (hb : b ≥ 2) (hc : c ≥ 2) :
  Real.log a^2 / Real.log (b + c) + Real.log b^2 / Real.log (a + c) + Real.log c^2 / Real.log (a + b) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_inequality_l2838_283835


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l2838_283877

def vec_a : ℝ × ℝ := (4, 2)

theorem opposite_unit_vector :
  let opposite_unit := (-vec_a.1 / Real.sqrt (vec_a.1^2 + vec_a.2^2),
                        -vec_a.2 / Real.sqrt (vec_a.1^2 + vec_a.2^2))
  opposite_unit = (-2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5) :=
by sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l2838_283877


namespace NUMINAMATH_CALUDE_rational_square_sum_l2838_283855

theorem rational_square_sum (a b c : ℚ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  ∃ r : ℚ, (1 / (a - b)^2 + 1 / (b - c)^2 + 1 / (c - a)^2) = r^2 := by
  sorry

end NUMINAMATH_CALUDE_rational_square_sum_l2838_283855


namespace NUMINAMATH_CALUDE_solution_to_sqrt_equation_l2838_283841

theorem solution_to_sqrt_equation :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
sorry

end NUMINAMATH_CALUDE_solution_to_sqrt_equation_l2838_283841


namespace NUMINAMATH_CALUDE_school_trip_buses_l2838_283864

/-- The number of buses needed for a school trip -/
def buses_needed (students : ℕ) (seats_per_bus : ℕ) : ℕ :=
  (students + seats_per_bus - 1) / seats_per_bus

/-- Proof that 5 buses are needed for 45 students with 9 seats per bus -/
theorem school_trip_buses :
  buses_needed 45 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_school_trip_buses_l2838_283864


namespace NUMINAMATH_CALUDE_drama_club_two_skills_l2838_283820

/-- Represents the number of students with a particular combination of skills -/
structure SkillCount where
  write : Nat
  direct : Nat
  produce : Nat
  write_direct : Nat
  write_produce : Nat
  direct_produce : Nat

/-- Represents the constraints of the drama club problem -/
def drama_club_constraints (sc : SkillCount) : Prop :=
  sc.write + sc.direct + sc.produce + sc.write_direct + sc.write_produce + sc.direct_produce = 150 ∧
  sc.write + sc.write_direct + sc.write_produce = 90 ∧
  sc.direct + sc.write_direct + sc.direct_produce = 60 ∧
  sc.produce + sc.write_produce + sc.direct_produce = 110

/-- The main theorem stating that under the given constraints, 
    the number of students with exactly two skills is 110 -/
theorem drama_club_two_skills (sc : SkillCount) 
  (h : drama_club_constraints sc) : 
  sc.write_direct + sc.write_produce + sc.direct_produce = 110 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_two_skills_l2838_283820


namespace NUMINAMATH_CALUDE_shoe_shirt_cost_difference_is_three_l2838_283831

/-- The cost difference between a pair of shoes and a shirt -/
def shoe_shirt_cost_difference : ℝ :=
  let shirt_cost : ℝ := 7
  let shoe_cost : ℝ := shirt_cost + shoe_shirt_cost_difference
  let bag_cost : ℝ := (2 * shirt_cost + shoe_cost) / 2
  let total_cost : ℝ := 2 * shirt_cost + shoe_cost + bag_cost
  shoe_shirt_cost_difference

/-- Theorem stating the cost difference between a pair of shoes and a shirt -/
theorem shoe_shirt_cost_difference_is_three :
  shoe_shirt_cost_difference = 3 := by
  sorry

#eval shoe_shirt_cost_difference

end NUMINAMATH_CALUDE_shoe_shirt_cost_difference_is_three_l2838_283831


namespace NUMINAMATH_CALUDE_subject_choice_theorem_l2838_283814

/-- The number of subjects available --/
def num_subjects : ℕ := 7

/-- The number of subjects each student must choose --/
def subjects_to_choose : ℕ := 3

/-- The number of ways Student A can choose subjects --/
def ways_for_A : ℕ := Nat.choose (num_subjects - 1) (subjects_to_choose - 1)

/-- The probability that both Students B and C choose physics --/
def prob_B_and_C_physics : ℚ := 
  (Nat.choose (num_subjects - 1) (subjects_to_choose - 1) ^ 2 : ℚ) / 
  (Nat.choose num_subjects subjects_to_choose ^ 2 : ℚ)

theorem subject_choice_theorem : 
  ways_for_A = 15 ∧ prob_B_and_C_physics = 9 / 49 := by sorry

end NUMINAMATH_CALUDE_subject_choice_theorem_l2838_283814


namespace NUMINAMATH_CALUDE_blocks_used_for_tower_l2838_283829

/-- Given Randy's block usage, prove the number of blocks used for the tower -/
theorem blocks_used_for_tower 
  (total_blocks : ℕ) 
  (blocks_for_house : ℕ) 
  (blocks_for_tower : ℕ) 
  (h1 : total_blocks = 95) 
  (h2 : blocks_for_house = 20) 
  (h3 : blocks_for_tower = blocks_for_house + 30) : 
  blocks_for_tower = 50 := by
  sorry

end NUMINAMATH_CALUDE_blocks_used_for_tower_l2838_283829


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2838_283866

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a + 2 = 0) → 
  (b^3 - 2*b + 2 = 0) → 
  (c^3 - 2*c + 2 = 0) → 
  (1/(a+1) + 1/(b+1) + 1/(c+1) = -1) := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2838_283866


namespace NUMINAMATH_CALUDE_base4_division_l2838_283815

/-- Represents a number in base 4 --/
def Base4 : Type := Nat

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : Nat) : Base4 := sorry

/-- Performs division in base 4 --/
def divBase4 (a b : Base4) : Base4 := sorry

/-- The theorem to be proved --/
theorem base4_division :
  divBase4 (toBase4 2023) (toBase4 13) = toBase4 155 := by sorry

end NUMINAMATH_CALUDE_base4_division_l2838_283815


namespace NUMINAMATH_CALUDE_complement_of_A_l2838_283896

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem complement_of_A : 
  (Set.univ : Set ℝ) \ A = {x : ℝ | x < -1 ∨ 2 ≤ x} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2838_283896


namespace NUMINAMATH_CALUDE_sin_translation_equivalence_l2838_283836

theorem sin_translation_equivalence :
  ∀ x : ℝ, 2 * Real.sin (3 * x + π / 6) = 2 * Real.sin (3 * (x + π / 18)) :=
by sorry

end NUMINAMATH_CALUDE_sin_translation_equivalence_l2838_283836


namespace NUMINAMATH_CALUDE_ultra_marathon_average_time_l2838_283868

/-- Calculates the average time per mile given the total distance and time -/
def averageTimePerMile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : ℚ :=
  let totalMinutes : ℕ := hours * 60 + minutes
  (totalMinutes : ℚ) / distance

theorem ultra_marathon_average_time :
  averageTimePerMile 32 4 52 = 9.125 := by
  sorry

end NUMINAMATH_CALUDE_ultra_marathon_average_time_l2838_283868


namespace NUMINAMATH_CALUDE_percentage_subtraction_l2838_283886

theorem percentage_subtraction (a : ℝ) : a - 0.02 * a = 0.98 * a := by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_l2838_283886


namespace NUMINAMATH_CALUDE_car_rental_theorem_l2838_283879

/-- Represents a car rental company's pricing model -/
structure CarRental where
  totalVehicles : ℕ
  baseRentalFee : ℕ
  feeIncrement : ℕ
  rentedMaintCost : ℕ
  nonRentedMaintCost : ℕ

/-- Calculates the number of rented vehicles given a rental fee -/
def rentedVehicles (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  cr.totalVehicles - (rentalFee - cr.baseRentalFee) / cr.feeIncrement

/-- Calculates the monthly revenue given a rental fee -/
def monthlyRevenue (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  let rented := rentedVehicles cr rentalFee
  rentalFee * rented - cr.rentedMaintCost * rented - cr.nonRentedMaintCost * (cr.totalVehicles - rented)

/-- The main theorem about the car rental company -/
theorem car_rental_theorem (cr : CarRental) 
    (h1 : cr.totalVehicles = 100)
    (h2 : cr.baseRentalFee = 3000)
    (h3 : cr.feeIncrement = 60)
    (h4 : cr.rentedMaintCost = 160)
    (h5 : cr.nonRentedMaintCost = 60) :
  (rentedVehicles cr 3900 = 85) ∧
  (∃ maxRevenue : ℕ, maxRevenue = 324040 ∧ 
    ∀ fee, monthlyRevenue cr fee ≤ maxRevenue) ∧
  (∃ maxFee : ℕ, maxFee = 4560 ∧
    monthlyRevenue cr maxFee = 324040 ∧
    ∀ fee, monthlyRevenue cr fee ≤ monthlyRevenue cr maxFee) :=
  sorry


end NUMINAMATH_CALUDE_car_rental_theorem_l2838_283879


namespace NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l2838_283856

/-- The equation of a circle with center (a, b) and radius r is (x-a)^2 + (y-b)^2 = r^2 -/
def CircleEquation (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

/-- A line is tangent to a circle if the distance from the center of the circle to the line equals the radius of the circle -/
def IsTangentLine (a b r : ℝ) (m n c : ℝ) : Prop :=
  r = |m*a + n*b + c| / Real.sqrt (m^2 + n^2)

/-- The theorem stating that (x-2)^2 + (y+1)^2 = 8 is the equation of the circle with center (2, -1) tangent to the line x + y = 5 -/
theorem circle_equation_with_tangent_line :
  ∀ x y : ℝ,
  CircleEquation 2 (-1) (Real.sqrt 8) x y ↔ 
  IsTangentLine 2 (-1) (Real.sqrt 8) 1 1 (-5) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_with_tangent_line_l2838_283856


namespace NUMINAMATH_CALUDE_substitution_sequences_remainder_l2838_283818

/-- Represents the number of possible substitution sequences in a basketball game -/
def substitutionSequences (totalPlayers startingPlayers maxSubstitutions : ℕ) : ℕ :=
  let substitutes := totalPlayers - startingPlayers
  let a0 := 1  -- No substitutions
  let a1 := startingPlayers * substitutes  -- One substitution
  let a2 := a1 * (startingPlayers - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (startingPlayers - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (startingPlayers - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- The main theorem stating the remainder of substitution sequences divided by 100 -/
theorem substitution_sequences_remainder :
  substitutionSequences 15 5 4 % 100 = 51 := by
  sorry


end NUMINAMATH_CALUDE_substitution_sequences_remainder_l2838_283818


namespace NUMINAMATH_CALUDE_pi_sixth_to_degrees_l2838_283876

theorem pi_sixth_to_degrees :
  ∀ (π : ℝ), π > 0 → (π / 6) * (180 / π) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_pi_sixth_to_degrees_l2838_283876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2838_283865

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- Theorem: In an arithmetic sequence with positive terms, 
    if a₂ = 1 - a₁ and a₄ = 9 - a₃, then a₄ + a₅ = 27 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
  (h1 : seq.a 2 = 1 - seq.a 1)
  (h2 : seq.a 4 = 9 - seq.a 3) :
  seq.a 4 + seq.a 5 = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2838_283865


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2838_283828

theorem geometric_series_sum :
  let a : ℕ := 1  -- first term
  let r : ℕ := 3  -- common ratio
  let n : ℕ := 8  -- number of terms
  let series_sum := (a * (r^n - 1)) / (r - 1)
  series_sum = 3280 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2838_283828


namespace NUMINAMATH_CALUDE_g_6_equals_666_l2838_283824

def g (x : ℝ) : ℝ := 3*x^4 - 19*x^3 + 31*x^2 - 27*x - 72

theorem g_6_equals_666 : g 6 = 666 := by
  sorry

end NUMINAMATH_CALUDE_g_6_equals_666_l2838_283824


namespace NUMINAMATH_CALUDE_trigonometric_equality_quadratic_equation_l2838_283854

theorem trigonometric_equality (x : ℝ) : 
  (1 - 2 * Real.sin x * Real.cos x) / (Real.cos x^2 - Real.sin x^2) = 
  (1 - Real.tan x) / (1 + Real.tan x) := by sorry

theorem quadratic_equation (θ a b : ℝ) :
  Real.tan θ + Real.sin θ = a ∧ Real.tan θ - Real.sin θ = b →
  (a^2 - b^2)^2 = 16 * a * b := by sorry

end NUMINAMATH_CALUDE_trigonometric_equality_quadratic_equation_l2838_283854


namespace NUMINAMATH_CALUDE_ellipse_properties_l2838_283891

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define eccentricity
def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - y + Real.sqrt 2 = 0

-- Define the theorem
theorem ellipse_properties
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = Real.sqrt 3 / 2)
  (h_tangent : ∃ (x y : ℝ), x^2 + y^2 = b^2 ∧ tangent_line x y) :
  -- 1. Equation of C
  (∀ x y, ellipse a b x y ↔ x^2/4 + y^2 = 1) ∧
  -- 2. Range of slope k
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    -Real.sqrt 3 / 6 < k ∧ k < Real.sqrt 3 / 6) ∧
  -- 3. Fixed intersection point
  (∀ M N E : ℝ × ℝ,
    ellipse a b M.1 M.2 →
    ellipse a b N.1 N.2 →
    M.1 = N.1 ∧ M.2 = -N.2 →
    M ≠ N →
    ellipse a b E.1 E.2 →
    (∃ k : ℝ, k ≠ 0 ∧ 
      N.2 - 0 = k * (N.1 - 4) ∧
      E.2 - 0 = k * (E.1 - 4)) →
    ∃ t : ℝ, M.2 - E.2 = ((E.2 + M.2) / (E.1 - M.1)) * (t - E.1) ∧ t = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ellipse_properties_l2838_283891


namespace NUMINAMATH_CALUDE_prime_sum_2019_power_l2838_283880

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_2019_power_l2838_283880


namespace NUMINAMATH_CALUDE_transformed_function_theorem_l2838_283805

def original_function (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

def rotate_180_degrees (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

def translate_upwards (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => f x + units

theorem transformed_function_theorem :
  (translate_upwards (rotate_180_degrees original_function) 3) = λ x => -2 * x^2 - 4 * x :=
by sorry

end NUMINAMATH_CALUDE_transformed_function_theorem_l2838_283805


namespace NUMINAMATH_CALUDE_system_solution_existence_l2838_283804

theorem system_solution_existence (a : ℝ) : 
  (∃ (b x y : ℝ), y = x^2 + a ∧ x^2 + y^2 + 2*b^2 = 2*b*(x - y) + 1) ↔ 
  a ≤ Real.sqrt 2 + 1/4 := by
sorry

end NUMINAMATH_CALUDE_system_solution_existence_l2838_283804


namespace NUMINAMATH_CALUDE_disjoint_subsets_bound_l2838_283873

theorem disjoint_subsets_bound (m : ℕ) (A B : Finset ℕ) : 
  A ⊆ Finset.range m → 
  B ⊆ Finset.range m → 
  A ∩ B = ∅ → 
  A.sum id = B.sum id → 
  (A.card : ℝ) < m / Real.sqrt 2 ∧ (B.card : ℝ) < m / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subsets_bound_l2838_283873


namespace NUMINAMATH_CALUDE_seven_keys_three_adjacent_l2838_283894

/-- The number of distinct arrangements of keys on a keychain. -/
def keychain_arrangements (total_keys : ℕ) (adjacent_keys : ℕ) : ℕ :=
  (adjacent_keys.factorial * ((total_keys - adjacent_keys + 1 - 1).factorial / 2))

/-- Theorem stating the number of distinct arrangements for 7 keys with 3 adjacent -/
theorem seven_keys_three_adjacent :
  keychain_arrangements 7 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_seven_keys_three_adjacent_l2838_283894


namespace NUMINAMATH_CALUDE_dans_work_time_l2838_283888

/-- Dan's work rate in job completion per hour -/
def dans_rate : ℚ := 1 / 15

/-- Annie's work rate in job completion per hour -/
def annies_rate : ℚ := 1 / 10

/-- The time Annie works to complete the job after Dan stops -/
def annies_time : ℚ := 6

theorem dans_work_time (x : ℚ) : 
  x * dans_rate + annies_time * annies_rate = 1 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_dans_work_time_l2838_283888


namespace NUMINAMATH_CALUDE_expected_asthma_cases_l2838_283890

theorem expected_asthma_cases (total_sample : ℕ) (asthma_rate : ℚ) 
  (h1 : total_sample = 320) 
  (h2 : asthma_rate = 1 / 8) : 
  ⌊total_sample * asthma_rate⌋ = 40 := by
  sorry

end NUMINAMATH_CALUDE_expected_asthma_cases_l2838_283890


namespace NUMINAMATH_CALUDE_line_direction_vector_l2838_283852

/-- Given two points and a direction vector, prove the value of b -/
theorem line_direction_vector (p1 p2 : ℝ × ℝ) (b : ℝ) :
  p1 = (-3, 2) →
  p2 = (4, -3) →
  ∃ (k : ℝ), k • (p2.1 - p1.1, p2.2 - p1.2) = (b, -2) →
  b = 14/5 := by
sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2838_283852


namespace NUMINAMATH_CALUDE_smallest_valid_n_l2838_283823

def is_valid_arrangement (n : ℕ) (a : Fin 9 → ℕ) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧
  (∀ i j, (i.val - j.val) % 9 ≥ 2 → n ∣ a i * a j) ∧
  (∀ i, ¬(n ∣ a i * a (i + 1)))

theorem smallest_valid_n : 
  (∃ (a : Fin 9 → ℕ), is_valid_arrangement 485100 a) ∧
  (∀ n < 485100, ¬∃ (a : Fin 9 → ℕ), is_valid_arrangement n a) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_n_l2838_283823


namespace NUMINAMATH_CALUDE_min_score_is_39_l2838_283826

/-- Represents the player's scores in a basketball season --/
structure BasketballScores where
  scores_6_to_9 : Fin 4 → ℕ
  avg_after_5 : ℝ
  avg_after_9 : ℝ

/-- The minimum score needed in the 10th game --/
def min_score_10th_game (bs : BasketballScores) : ℕ :=
  sorry

/-- Theorem stating the minimum score in the 10th game is 39 --/
theorem min_score_is_39 (bs : BasketballScores) : 
  (bs.scores_6_to_9 0 = 25) →
  (bs.scores_6_to_9 1 = 14) →
  (bs.scores_6_to_9 2 = 15) →
  (bs.scores_6_to_9 3 = 22) →
  (16 < bs.avg_after_5) →
  (bs.avg_after_5 < 17) →
  (bs.avg_after_5 < bs.avg_after_9) →
  (min_score_10th_game bs = 39) :=
  sorry

end NUMINAMATH_CALUDE_min_score_is_39_l2838_283826


namespace NUMINAMATH_CALUDE_max_students_above_mean_l2838_283833

/-- Given a class of 150 students, proves that the maximum number of students
    who can have a score higher than the class mean is 149. -/
theorem max_students_above_mean (scores : Fin 150 → ℝ) :
  (Finset.filter (fun i => scores i > Finset.sum Finset.univ scores / 150) Finset.univ).card ≤ 149 :=
by
  sorry

end NUMINAMATH_CALUDE_max_students_above_mean_l2838_283833


namespace NUMINAMATH_CALUDE_cookie_problem_l2838_283822

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l2838_283822


namespace NUMINAMATH_CALUDE_xiaoming_money_l2838_283889

/-- Proves that Xiaoming brought 108 yuan to the supermarket -/
theorem xiaoming_money (fresh_milk_cost yogurt_cost : ℕ) 
  (fresh_milk_cartons yogurt_cartons total_money : ℕ) : 
  fresh_milk_cost = 6 →
  yogurt_cost = 9 →
  fresh_milk_cost * fresh_milk_cartons = total_money →
  yogurt_cost * yogurt_cartons = total_money →
  fresh_milk_cartons = yogurt_cartons + 6 →
  total_money = 108 := by
  sorry

#check xiaoming_money

end NUMINAMATH_CALUDE_xiaoming_money_l2838_283889


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2838_283802

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l2838_283802


namespace NUMINAMATH_CALUDE_roundness_of_eight_million_l2838_283858

/-- Roundness of a positive integer is the sum of exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million :
  roundness 8000000 = 15 := by sorry

end NUMINAMATH_CALUDE_roundness_of_eight_million_l2838_283858


namespace NUMINAMATH_CALUDE_part_one_part_two_l2838_283881

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  -- Add necessary conditions
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  pos_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π
  -- Add the given condition
  given_condition : 2 * Real.cos C + 2 * Real.cos A = 5 * b / 2

theorem part_one (t : Triangle) : 2 * (t.a + t.c) = 3 * t.b := by sorry

theorem part_two (t : Triangle) (h1 : Real.cos t.B = 1/4) (h2 : t.S = Real.sqrt 15) : t.b = 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2838_283881


namespace NUMINAMATH_CALUDE_kate_pen_purchase_l2838_283832

theorem kate_pen_purchase (pen_cost : ℝ) (kate_money : ℝ) : 
  pen_cost = 30 → kate_money = pen_cost / 3 → pen_cost - kate_money = 20 := by
  sorry

end NUMINAMATH_CALUDE_kate_pen_purchase_l2838_283832


namespace NUMINAMATH_CALUDE_trash_outside_classrooms_l2838_283870

theorem trash_outside_classrooms 
  (total_trash : ℕ) 
  (classroom_trash : ℕ) 
  (h1 : total_trash = 1576) 
  (h2 : classroom_trash = 344) : 
  total_trash - classroom_trash = 1232 := by
sorry

end NUMINAMATH_CALUDE_trash_outside_classrooms_l2838_283870


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2838_283857

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2838_283857


namespace NUMINAMATH_CALUDE_cos_double_angle_special_case_l2838_283807

theorem cos_double_angle_special_case (θ : Real) 
  (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_case_l2838_283807


namespace NUMINAMATH_CALUDE_geraldine_doll_count_l2838_283887

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.0

/-- The number of additional dolls Geraldine has compared to Jazmin -/
def additional_dolls : ℕ := 977

/-- The total number of dolls Geraldine has -/
def geraldine_dolls : ℝ := jazmin_dolls + additional_dolls

theorem geraldine_doll_count : geraldine_dolls = 2186 := by
  sorry

end NUMINAMATH_CALUDE_geraldine_doll_count_l2838_283887


namespace NUMINAMATH_CALUDE_factors_of_243_l2838_283813

theorem factors_of_243 : Finset.card (Nat.divisors 243) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_l2838_283813


namespace NUMINAMATH_CALUDE_solve_system_and_calculate_l2838_283819

theorem solve_system_and_calculate (x y : ℚ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : x + 2 * y = 10) : 
  (x + y) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_solve_system_and_calculate_l2838_283819


namespace NUMINAMATH_CALUDE_inequality_proof_l2838_283843

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hsum : a + b + c = 1) :
  a^(a^2 + 2*c*a) * b^(b^2 + 2*a*b) * c^(c^2 + 2*b*c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2838_283843


namespace NUMINAMATH_CALUDE_train_distance_problem_l2838_283817

/-- Proves that the distance between two stations is 540 km given the conditions of the train problem -/
theorem train_distance_problem (v1 v2 : ℝ) (d : ℝ) :
  v1 = 20 →  -- Speed of train 1 in km/hr
  v2 = 25 →  -- Speed of train 2 in km/hr
  v2 > v1 →  -- Train 2 is faster than train 1
  d = (v2 - v1) * (v1 * v2)⁻¹ * 60 →  -- Difference in distance traveled
  v1 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) + 
  v2 * ((v1 + v2) * (v2 - v1)⁻¹ * 60) = 540 :=
by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2838_283817


namespace NUMINAMATH_CALUDE_baba_yaga_students_l2838_283895

theorem baba_yaga_students (B G : ℕ) : 
  B + G = 33 →
  (2 * G + 2 * B) / 3 = 22 :=
by
  sorry

#check baba_yaga_students

end NUMINAMATH_CALUDE_baba_yaga_students_l2838_283895


namespace NUMINAMATH_CALUDE_smallest_k_for_difference_property_l2838_283899

theorem smallest_k_for_difference_property (n : ℕ) (hn : n ≥ 1) :
  let k := n^2 + 2
  ∀ (S : Finset ℝ), S.card ≥ k →
    ∃ (x y : ℝ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧
      (|x - y| < 1 / n ∨ |x - y| > n) ∧
    ∀ (m : ℕ), m < k →
      ∃ (T : Finset ℝ), T.card = m ∧
        ∀ (a b : ℝ), a ∈ T ∧ b ∈ T ∧ a ≠ b →
          |a - b| ≥ 1 / n ∧ |a - b| ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_difference_property_l2838_283899


namespace NUMINAMATH_CALUDE_ellipse_m_relation_l2838_283803

/-- Represents an ellipse with parameter m -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_distance : ℝ

/-- The theorem stating the relationship between m and the focal distance -/
theorem ellipse_m_relation (m : ℝ) (e : Ellipse m) (h : e.focal_distance = 4) :
  16 = 2 * m - 12 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_m_relation_l2838_283803


namespace NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l2838_283806

def standard_deck : ℕ := 52
def hearts_and_kings : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - hearts_and_kings / standard_deck) ^ 2
  p = 88 / 169 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_heart_or_king_l2838_283806


namespace NUMINAMATH_CALUDE_supermarket_fruit_prices_l2838_283846

theorem supermarket_fruit_prices 
  (strawberry_pints : ℕ) 
  (strawberry_sale_revenue : ℕ) 
  (strawberry_revenue_difference : ℕ)
  (blueberry_pints : ℕ) 
  (blueberry_sale_revenue : ℕ) 
  (blueberry_revenue_difference : ℕ)
  (h1 : strawberry_pints = 54)
  (h2 : strawberry_sale_revenue = 216)
  (h3 : strawberry_revenue_difference = 108)
  (h4 : blueberry_pints = 36)
  (h5 : blueberry_sale_revenue = 144)
  (h6 : blueberry_revenue_difference = 72) :
  (strawberry_sale_revenue + strawberry_revenue_difference) / strawberry_pints = 
  (blueberry_sale_revenue + blueberry_revenue_difference) / blueberry_pints :=
by sorry

end NUMINAMATH_CALUDE_supermarket_fruit_prices_l2838_283846


namespace NUMINAMATH_CALUDE_floor_inequality_l2838_283871

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_inequality_l2838_283871


namespace NUMINAMATH_CALUDE_value_of_x_l2838_283882

theorem value_of_x (x y z a b c : ℝ) 
  (ha : x * y / (x + y) = a)
  (hb : x * z / (x + z) = b)
  (hc : y * z / (y + z) = c)
  (ha_nonzero : a ≠ 0)
  (hb_nonzero : b ≠ 0)
  (hc_nonzero : c ≠ 0) :
  x = 2 * a * b * c / (a * c + b * c - a * b) :=
by sorry

end NUMINAMATH_CALUDE_value_of_x_l2838_283882


namespace NUMINAMATH_CALUDE_race_probability_l2838_283883

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) :
  total_cars = 15 →
  prob_x = 1 / 4 →
  prob_y = 1 / 8 →
  prob_z = 1 / 12 →
  (prob_x + prob_y + prob_z : ℚ) = 11 / 24 :=
by sorry

end NUMINAMATH_CALUDE_race_probability_l2838_283883


namespace NUMINAMATH_CALUDE_sum_of_squared_ratios_bound_l2838_283810

/-- Given positive real numbers a, b, and c, 
    the sum of three terms in the form (2x+y+z)²/(2x²+(y+z)²) 
    where x, y, z are cyclic permutations of a, b, c, is less than or equal to 8 -/
theorem sum_of_squared_ratios_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (2*a + b + c)^2 / (2*a^2 + (b + c)^2) + 
  (2*b + a + c)^2 / (2*b^2 + (c + a)^2) + 
  (2*c + a + b)^2 / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_ratios_bound_l2838_283810


namespace NUMINAMATH_CALUDE_rectangle_diagonal_rectangle_diagonal_proof_l2838_283827

/-- The length of the diagonal of a rectangle with length 100 and width 100√2 is 100√3 -/
theorem rectangle_diagonal : Real → Prop :=
  fun d =>
    let length := 100
    let width := 100 * Real.sqrt 2
    d = 100 * Real.sqrt 3 ∧ d^2 = length^2 + width^2

/-- Proof of the theorem -/
theorem rectangle_diagonal_proof : ∃ d, rectangle_diagonal d := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_rectangle_diagonal_proof_l2838_283827


namespace NUMINAMATH_CALUDE_resulting_figure_sides_l2838_283800

/-- Represents a polygon in the construction --/
structure Polygon :=
  (sides : ℕ)
  (adjacentSides : ℕ)

/-- The construction of polygons --/
def construction : List Polygon :=
  [{ sides := 3, adjacentSides := 1 },  -- isosceles triangle
   { sides := 4, adjacentSides := 2 },  -- rectangle
   { sides := 6, adjacentSides := 2 },  -- first hexagon
   { sides := 7, adjacentSides := 2 },  -- heptagon
   { sides := 6, adjacentSides := 2 },  -- second hexagon
   { sides := 9, adjacentSides := 1 }]  -- nonagon

theorem resulting_figure_sides :
  (construction.map (λ p => p.sides - p.adjacentSides)).sum = 25 := by
  sorry

end NUMINAMATH_CALUDE_resulting_figure_sides_l2838_283800


namespace NUMINAMATH_CALUDE_bumper_car_line_l2838_283848

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := sorry

/-- The number of people in line after 2 leave and 2 join -/
def final_people : ℕ := 10

/-- The condition that if 2 people leave and 2 join, there are 10 people in line -/
axiom condition : initial_people = final_people

theorem bumper_car_line : initial_people = 10 := by sorry

end NUMINAMATH_CALUDE_bumper_car_line_l2838_283848
