import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_area_l3454_345429

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 100) (hb : b^2 = 64) (hc : c^2 = 121)
  (h_right : a^2 + b^2 = c^2) : (1/2) * a * b = 40 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3454_345429


namespace NUMINAMATH_CALUDE_initial_value_theorem_l3454_345498

theorem initial_value_theorem (y : ℕ) (h : y > 0) :
  ∃ x : ℤ, (x : ℤ) + 49 = y^2 ∧ x = y^2 - 49 :=
by sorry

end NUMINAMATH_CALUDE_initial_value_theorem_l3454_345498


namespace NUMINAMATH_CALUDE_arccos_cos_eleven_l3454_345436

theorem arccos_cos_eleven : 
  Real.arccos (Real.cos 11) = 11 - 3 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_eleven_l3454_345436


namespace NUMINAMATH_CALUDE_triangle_area_l3454_345476

/-- Given a triangle ABC with the following properties:
  * sinB = √2 * sinA
  * ∠C = 105°
  * c = √3 + 1
  Prove that the area of triangle ABC is (√3 + 1) / 2 -/
theorem triangle_area (A B C : ℝ) (h1 : Real.sin B = Real.sqrt 2 * Real.sin A)
  (h2 : C = 105 * π / 180) (h3 : Real.sqrt 3 + 1 = 2 * Real.sin (C / 2) * Real.sin ((A + B) / 2)) :
  (Real.sqrt 3 + 1) / 2 = (Real.sin C) * (Real.sin A) * (Real.sin B) / (Real.sin (A + B)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3454_345476


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3454_345496

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℝ), 
  a = 80 → 
  b = 150 → 
  c^2 = a^2 + b^2 → 
  c = 170 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3454_345496


namespace NUMINAMATH_CALUDE_abc_condition_neither_sufficient_nor_necessary_l3454_345475

theorem abc_condition_neither_sufficient_nor_necessary :
  ¬ (∀ a b c : ℝ, a * b * c = 1 → 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c) ∧
  ¬ (∀ a b c : ℝ, 1 / Real.sqrt a + 1 / Real.sqrt b + 1 / Real.sqrt c ≤ a + b + c → a * b * c = 1) :=
by sorry

end NUMINAMATH_CALUDE_abc_condition_neither_sufficient_nor_necessary_l3454_345475


namespace NUMINAMATH_CALUDE_harriet_round_trip_l3454_345438

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_round_trip 
  (d : ℝ) -- distance between A-ville and B-town in km
  (speed_to_b : ℝ) -- speed from A-ville to B-town in km/h
  (time_to_b : ℝ) -- time taken from A-ville to B-town in hours
  (total_time : ℝ) -- total round trip time in hours
  (h1 : d = speed_to_b * time_to_b) -- distance = speed * time for A-ville to B-town
  (h2 : speed_to_b = 100) -- speed from A-ville to B-town is 100 km/h
  (h3 : time_to_b = 3) -- time taken from A-ville to B-town is 3 hours
  (h4 : total_time = 5) -- total round trip time is 5 hours
  : d / (total_time - time_to_b) = 150 := by
  sorry

end NUMINAMATH_CALUDE_harriet_round_trip_l3454_345438


namespace NUMINAMATH_CALUDE_problem_p5_l3454_345454

theorem problem_p5 (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h1 : a^2 + b^2 = 1)
  (h2 : c^2 + d^2 = 1)
  (h3 : a * d - b * c = 1/7) :
  a * c + b * d = 4 * Real.sqrt 3 / 7 := by
sorry

end NUMINAMATH_CALUDE_problem_p5_l3454_345454


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3454_345439

theorem mindmaster_codes (num_slots : ℕ) (num_colors : ℕ) : 
  num_slots = 5 → num_colors = 7 → num_colors ^ num_slots = 16807 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3454_345439


namespace NUMINAMATH_CALUDE_complex_real_condition_l3454_345404

theorem complex_real_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a^2 - 1) : ℂ) + (Complex.I : ℂ) * (a + 1)).im = 0 →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3454_345404


namespace NUMINAMATH_CALUDE_ruby_height_l3454_345478

/-- Given the heights of various people, prove Ruby's height -/
theorem ruby_height
  (janet_height : ℕ)
  (charlene_height : ℕ)
  (pablo_height : ℕ)
  (ruby_height : ℕ)
  (h1 : janet_height = 62)
  (h2 : charlene_height = 2 * janet_height)
  (h3 : pablo_height = charlene_height + 70)
  (h4 : ruby_height = pablo_height - 2)
  : ruby_height = 192 := by
  sorry

end NUMINAMATH_CALUDE_ruby_height_l3454_345478


namespace NUMINAMATH_CALUDE_g_range_l3454_345485

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 3 * Real.cos x ^ 2 - 4 * Real.cos x + 5 * Real.sin x ^ 2 - 7) / (Real.cos x - 2)

theorem g_range : 
  ∀ y : ℝ, (∃ x : ℝ, Real.cos x ≠ 2 ∧ g x = y) ↔ 1 ≤ y ∧ y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_g_range_l3454_345485


namespace NUMINAMATH_CALUDE_words_with_e_count_l3454_345474

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding E -/
def alphabet_size_without_e : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words without E -/
def words_without_e : ℕ := alphabet_size_without_e ^ word_length

/-- The number of words with at least one E -/
def words_with_e : ℕ := total_words - words_without_e

theorem words_with_e_count : words_with_e = 369 := by
  sorry

end NUMINAMATH_CALUDE_words_with_e_count_l3454_345474


namespace NUMINAMATH_CALUDE_wednesday_bags_is_nine_l3454_345441

/-- Represents the leaf raking business scenario -/
structure LeafRakingBusiness where
  charge_per_bag : ℕ
  monday_bags : ℕ
  tuesday_bags : ℕ
  total_earnings : ℕ

/-- Calculates the number of bags raked on Wednesday -/
def bags_on_wednesday (business : LeafRakingBusiness) : ℕ :=
  (business.total_earnings - business.charge_per_bag * (business.monday_bags + business.tuesday_bags)) / business.charge_per_bag

/-- Theorem stating that the number of bags raked on Wednesday is 9 -/
theorem wednesday_bags_is_nine (business : LeafRakingBusiness)
  (h1 : business.charge_per_bag = 4)
  (h2 : business.monday_bags = 5)
  (h3 : business.tuesday_bags = 3)
  (h4 : business.total_earnings = 68) :
  bags_on_wednesday business = 9 := by
  sorry

#eval bags_on_wednesday { charge_per_bag := 4, monday_bags := 5, tuesday_bags := 3, total_earnings := 68 }

end NUMINAMATH_CALUDE_wednesday_bags_is_nine_l3454_345441


namespace NUMINAMATH_CALUDE_banknote_sum_divisibility_l3454_345442

theorem banknote_sum_divisibility
  (a b : ℕ)
  (h_distinct : a % 101 ≠ b % 101)
  (h_total : ℕ)
  (h_count : h_total = 100) :
  ∃ (m n : ℕ), m + n ≤ h_total ∧ (m * a + n * b) % 101 = 0 :=
sorry

end NUMINAMATH_CALUDE_banknote_sum_divisibility_l3454_345442


namespace NUMINAMATH_CALUDE_metal_waste_l3454_345405

theorem metal_waste (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let circle_area := Real.pi * (s/2)^2
  let inner_square_side := s / Real.sqrt 2
  let inner_square_area := inner_square_side^2
  let waste := square_area - circle_area + (circle_area - inner_square_area)
  waste = square_area / 2 :=
by sorry

end NUMINAMATH_CALUDE_metal_waste_l3454_345405


namespace NUMINAMATH_CALUDE_vector_expressions_equal_AD_l3454_345432

variable {V : Type*} [AddCommGroup V]

variable (A B C D M O : V)

theorem vector_expressions_equal_AD :
  (A - D + M - B) + (B - C + C - M) = A - D ∧
  (A - B + C - D) + (B - C) = A - D ∧
  (O - C) - (O - A) + (C - D) = A - D :=
by sorry

end NUMINAMATH_CALUDE_vector_expressions_equal_AD_l3454_345432


namespace NUMINAMATH_CALUDE_attendance_difference_l3454_345452

/-- Proves that given the initial ratio of boys to girls to adults as 9.5:6.25:4.75,
    with 30% of attendees being girls, and after 15% of girls and 20% of adults leave,
    the percentage difference between boys and the combined number of remaining girls
    and adults is approximately 2.304%. -/
theorem attendance_difference (boys girls adults : ℝ) 
    (h_ratio : boys = 9.5 ∧ girls = 6.25 ∧ adults = 4.75)
    (h_girls_percent : girls / (boys + girls + adults) = 0.3)
    (h_girls_left : ℝ) (h_adults_left : ℝ)
    (h_girls_left_percent : h_girls_left = 0.15)
    (h_adults_left_percent : h_adults_left = 0.2) :
    let total := boys + girls + adults
    let boys_percent := boys / total
    let girls_adjusted := girls * (1 - h_girls_left)
    let adults_adjusted := adults * (1 - h_adults_left)
    let girls_adults_adjusted_percent := (girls_adjusted + adults_adjusted) / total
    abs (boys_percent - girls_adults_adjusted_percent - 0.02304) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_attendance_difference_l3454_345452


namespace NUMINAMATH_CALUDE_cos_2x_plus_pi_third_equiv_sin_2x_shifted_l3454_345471

theorem cos_2x_plus_pi_third_equiv_sin_2x_shifted (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_plus_pi_third_equiv_sin_2x_shifted_l3454_345471


namespace NUMINAMATH_CALUDE_range_of_f_l3454_345409

def f (x : ℝ) : ℝ := x^2 - 6*x + 10

theorem range_of_f :
  Set.range f = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3454_345409


namespace NUMINAMATH_CALUDE_min_value_of_f_l3454_345486

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), m = 50/27 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3454_345486


namespace NUMINAMATH_CALUDE_grape_juice_in_drink_l3454_345408

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ

/-- Calculates the amount of grape juice in the drink -/
def grape_juice_amount (drink : FruitDrink) : ℝ :=
  drink.total * (1 - drink.orange_percent - drink.watermelon_percent)

/-- Theorem stating the amount of grape juice in the specific drink -/
theorem grape_juice_in_drink : 
  let drink : FruitDrink := { total := 150, orange_percent := 0.35, watermelon_percent := 0.35 }
  grape_juice_amount drink = 45 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_in_drink_l3454_345408


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_l3454_345468

/-- Given an ellipse ax^2 + by^2 = 1 intersecting with the line y = 1 - x,
    if the slope of the line passing through the origin and the midpoint
    of the intersection points is √3/2, then a/b = √3/2. -/
theorem ellipse_intersection_slope (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 + b * (1 - x₁)^2 = 1 ∧
                a * x₂^2 + b * (1 - x₂)^2 = 1 ∧
                x₁ ≠ x₂) →
  ((b / (a + b)) / (a / (a + b)) = Real.sqrt 3 / 2) →
  a / b = Real.sqrt 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_l3454_345468


namespace NUMINAMATH_CALUDE_set_operation_equality_l3454_345472

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define sets A and B
def A : Set Nat := {0, 3, 4}
def B : Set Nat := {1, 3}

-- State the theorem
theorem set_operation_equality :
  (Aᶜ ∪ A) ∪ B = {1, 2, 3} :=
by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l3454_345472


namespace NUMINAMATH_CALUDE_coprime_linear_combination_l3454_345470

theorem coprime_linear_combination (a b n : ℕ+) (h1 : Nat.Coprime a b) (h2 : n > a * b) :
  ∃ (x y : ℕ+), n = a * x + b * y := by
sorry

end NUMINAMATH_CALUDE_coprime_linear_combination_l3454_345470


namespace NUMINAMATH_CALUDE_circle_diameter_l3454_345449

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 100 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 20 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l3454_345449


namespace NUMINAMATH_CALUDE_linda_spent_correct_l3454_345487

/-- The total amount Linda spent on school supplies -/
def linda_total_spent : ℝ := 6.80

/-- The cost of a single notebook -/
def notebook_cost : ℝ := 1.20

/-- The number of notebooks Linda bought -/
def notebook_quantity : ℕ := 3

/-- The cost of a box of pencils -/
def pencil_box_cost : ℝ := 1.50

/-- The cost of a box of pens -/
def pen_box_cost : ℝ := 1.70

/-- Theorem stating that the total amount Linda spent is correct -/
theorem linda_spent_correct :
  linda_total_spent = notebook_cost * (notebook_quantity : ℝ) + pencil_box_cost + pen_box_cost := by
  sorry

end NUMINAMATH_CALUDE_linda_spent_correct_l3454_345487


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3454_345479

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat
  lastInningScore : Nat

/-- Calculates the new average of a batsman after an additional inning -/
def newAverage (performance : BatsmanPerformance) : Nat :=
  (performance.totalRuns + performance.lastInningScore) / (performance.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 85 in the 11th inning,
    then his new average is 35 -/
theorem batsman_average_theorem (performance : BatsmanPerformance) 
  (h1 : performance.innings = 10)
  (h2 : performance.lastInningScore = 85)
  (h3 : performance.averageIncrease = 5) :
  newAverage performance = 35 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l3454_345479


namespace NUMINAMATH_CALUDE_little_john_money_l3454_345419

/-- Little John's money problem -/
theorem little_john_money (sweet_cost : ℚ) (friend_gift : ℚ) (num_friends : ℕ) (money_left : ℚ) 
  (h1 : sweet_cost = 105/100)
  (h2 : friend_gift = 1)
  (h3 : num_friends = 2)
  (h4 : money_left = 205/100) :
  sweet_cost + num_friends * friend_gift + money_left = 51/10 := by
  sorry

end NUMINAMATH_CALUDE_little_john_money_l3454_345419


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l3454_345457

/-- The perimeter of an irregular pentagon with given side lengths is 52.9 cm -/
theorem pentagon_perimeter (s1 s2 s3 s4 s5 : ℝ) 
  (h1 : s1 = 5.2) (h2 : s2 = 10.3) (h3 : s3 = 15.8) (h4 : s4 = 8.7) (h5 : s5 = 12.9) :
  s1 + s2 + s3 + s4 + s5 = 52.9 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l3454_345457


namespace NUMINAMATH_CALUDE_smallest_divisible_term_l3454_345402

/-- Geometric sequence with first term a and common ratio r -/
def geometricSequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n - 1)

/-- The common ratio of the geometric sequence -/
def commonRatio : ℚ := 25 / (5/6)

/-- The nth term of the specific geometric sequence -/
def nthTerm (n : ℕ) : ℚ := geometricSequence (5/6) commonRatio n

/-- Predicate to check if a rational number is divisible by 2,000,000 -/
def divisibleByTwoMillion (q : ℚ) : Prop := ∃ (k : ℤ), q = (2000000 : ℚ) * k

/-- Statement: 8 is the smallest positive integer n such that the nth term 
    of the geometric sequence is divisible by 2,000,000 -/
theorem smallest_divisible_term : 
  (∀ m : ℕ, m < 8 → ¬(divisibleByTwoMillion (nthTerm m))) ∧ 
  (divisibleByTwoMillion (nthTerm 8)) := by sorry

end NUMINAMATH_CALUDE_smallest_divisible_term_l3454_345402


namespace NUMINAMATH_CALUDE_club_members_proof_l3454_345463

theorem club_members_proof (total : Nat) (left_handed : Nat) (rock_fans : Nat) (right_handed_non_rock : Nat) 
  (h1 : total = 30)
  (h2 : left_handed = 12)
  (h3 : rock_fans = 20)
  (h4 : right_handed_non_rock = 3)
  (h5 : ∀ x : Nat, x ≤ total → x = (left_handed + (total - left_handed)))
  : ∃ x : Nat, x = 5 ∧ 
    x + (left_handed - x) + (rock_fans - x) + right_handed_non_rock = total := by
  sorry


end NUMINAMATH_CALUDE_club_members_proof_l3454_345463


namespace NUMINAMATH_CALUDE_candy_bar_cost_candy_bar_cost_is_7_l3454_345483

def chocolate_cost : ℕ := 3
def extra_cost : ℕ := 4

theorem candy_bar_cost : ℕ :=
  chocolate_cost + extra_cost

#check candy_bar_cost

theorem candy_bar_cost_is_7 : candy_bar_cost = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_candy_bar_cost_is_7_l3454_345483


namespace NUMINAMATH_CALUDE_no_perfect_square_solution_l3454_345490

theorem no_perfect_square_solution : 
  ¬ ∃ (n : ℕ+) (m : ℕ), n^2 + 12*n - 2006 = m^2 := by
sorry

end NUMINAMATH_CALUDE_no_perfect_square_solution_l3454_345490


namespace NUMINAMATH_CALUDE_b_range_l3454_345417

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem b_range (a b : ℝ) (h : f a = g b) : 
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_b_range_l3454_345417


namespace NUMINAMATH_CALUDE_square_side_length_l3454_345421

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 225 → side * side = area → side = 15 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3454_345421


namespace NUMINAMATH_CALUDE_tv_price_change_l3454_345445

theorem tv_price_change (P : ℝ) : 
  P > 0 → (P * 0.8 * 1.45) = P * 1.16 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l3454_345445


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_count_l3454_345446

/-- The number of intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem stating that the number of interior intersection points of diagonals
    in a regular decagon is equal to the number of ways to choose 4 vertices from 10 -/
theorem decagon_diagonal_intersections_count :
  decagon_diagonal_intersections = 210 := by
  sorry

#eval decagon_diagonal_intersections

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_count_l3454_345446


namespace NUMINAMATH_CALUDE_power_multiplication_l3454_345451

theorem power_multiplication : 3^6 * 4^3 = 46656 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3454_345451


namespace NUMINAMATH_CALUDE_a_value_m_range_l3454_345437

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Theorem 1: Prove that a = 1
theorem a_value (a : ℝ) : 
  (∀ x, f a x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) → a = 1 := by sorry

-- Theorem 2: Prove that the minimum value of m is 4
theorem m_range : 
  ∃ m : ℝ, (∃ n : ℝ, f 1 n ≤ m - f 1 (-n)) ∧
  (∀ m' : ℝ, (∃ n : ℝ, f 1 n ≤ m' - f 1 (-n)) → m' ≥ m) ∧
  m = 4 := by sorry

end NUMINAMATH_CALUDE_a_value_m_range_l3454_345437


namespace NUMINAMATH_CALUDE_least_sum_m_n_l3454_345416

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val.gcd (330 : ℕ) = 1) ∧ 
  (n.val.gcd (330 : ℕ) = 1) ∧
  ((m + n).val.gcd (330 : ℕ) = 1) ∧
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧
  (∀ (l : ℕ+), m.val ≠ l.val * n.val) ∧
  (∀ (p q : ℕ+), 
    (p.val.gcd (330 : ℕ) = 1) ∧ 
    (q.val.gcd (330 : ℕ) = 1) ∧
    ((p + q).val.gcd (330 : ℕ) = 1) ∧
    (∃ (r : ℕ), p.val^p.val = r * q.val^q.val) ∧
    (∀ (s : ℕ+), p.val ≠ s.val * q.val) →
    (m + n).val ≤ (p + q).val) ∧
  (m + n).val = 154 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l3454_345416


namespace NUMINAMATH_CALUDE_f_f_10_equals_1_l3454_345464

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 10^(x-1) else Real.log x / Real.log 10

-- State the theorem
theorem f_f_10_equals_1 : f (f 10) = 1 := by sorry

end NUMINAMATH_CALUDE_f_f_10_equals_1_l3454_345464


namespace NUMINAMATH_CALUDE_nickels_remaining_l3454_345444

def initial_nickels : ℕ := 87
def borrowed_nickels : ℕ := 75

theorem nickels_remaining (initial : ℕ) (borrowed : ℕ) :
  initial ≥ borrowed → initial - borrowed = initial_nickels - borrowed_nickels :=
by sorry

end NUMINAMATH_CALUDE_nickels_remaining_l3454_345444


namespace NUMINAMATH_CALUDE_cape_may_has_24_sightings_l3454_345415

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 16

/-- The total number of shark sightings in Cape May and Daytona Beach -/
def total_sightings : ℕ := 40

/-- Theorem stating that Cape May has 24 shark sightings given the conditions -/
theorem cape_may_has_24_sightings :
  cape_may_sightings = 24 ∧
  cape_may_sightings + daytona_beach_sightings = total_sightings ∧
  cape_may_sightings = 2 * daytona_beach_sightings - 8 :=
by sorry

end NUMINAMATH_CALUDE_cape_may_has_24_sightings_l3454_345415


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l3454_345406

theorem opposite_sign_sum (a b : ℝ) : 
  (|a + 1| + |b + 2| = 0) → (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l3454_345406


namespace NUMINAMATH_CALUDE_composition_equation_solution_l3454_345460

theorem composition_equation_solution (p q : ℝ → ℝ) (c : ℝ) :
  (∀ x, p x = 3 * x - 8) →
  (∀ x, q x = 5 * x - c) →
  p (q 3) = 14 →
  c = 23 / 3 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l3454_345460


namespace NUMINAMATH_CALUDE_relationship_abc_l3454_345426

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := 2^(-1/3 : ℝ)
noncomputable def c : ℝ := Real.log 30 / Real.log 3

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3454_345426


namespace NUMINAMATH_CALUDE_even_product_probability_l3454_345467

def eight_sided_die := Finset.range 8

theorem even_product_probability :
  let outcomes := eight_sided_die.product eight_sided_die
  (outcomes.filter (fun (x, y) => (x + 1) * (y + 1) % 2 = 0)).card / outcomes.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_even_product_probability_l3454_345467


namespace NUMINAMATH_CALUDE_finite_subsequence_exists_infinite_subsequence_not_exists_l3454_345424

/-- The sequence 1, 1/2, 1/3, ... -/
def harmonic_sequence : ℕ → ℚ 
  | n => 1 / n

/-- A subsequence of the harmonic sequence -/
structure Subsequence :=
  (indices : ℕ → ℕ)
  (strictly_increasing : ∀ n, indices n < indices (n + 1))

/-- The property that each term from the third is the difference of the two preceding terms -/
def has_difference_property (s : Subsequence) : Prop :=
  ∀ k ≥ 3, harmonic_sequence (s.indices k) = 
    harmonic_sequence (s.indices (k - 2)) - harmonic_sequence (s.indices (k - 1))

theorem finite_subsequence_exists : ∃ s : Subsequence, 
  (∀ n, n ≤ 100 → s.indices n ≤ 100) ∧ has_difference_property s :=
sorry

theorem infinite_subsequence_not_exists : ¬∃ s : Subsequence, has_difference_property s :=
sorry

end NUMINAMATH_CALUDE_finite_subsequence_exists_infinite_subsequence_not_exists_l3454_345424


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3454_345453

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ α β : ℝ, 3 * α^2 - k * α + p = 0 ∧ 3 * β^2 - k * β + p = 0) →
  (∃ α β : ℝ, α + β = 9 ∧ α * β = 10) →
  k + p = 57 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l3454_345453


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_same_foci_l3454_345480

/-- Given a hyperbola and an ellipse with the same foci, prove that m = 1/11 -/
theorem hyperbola_ellipse_same_foci (m : ℝ) : 
  (∃ (c : ℝ), c^2 = 2*m ∧ c^2 = (m+1)/6) → m = 1/11 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_same_foci_l3454_345480


namespace NUMINAMATH_CALUDE_profit_increase_l3454_345440

theorem profit_increase (x : ℝ) : 
  (1 + x / 100) * 0.8 * 1.5 = 1.68 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_increase_l3454_345440


namespace NUMINAMATH_CALUDE_game_ends_in_three_rounds_l3454_345423

/-- Represents a player in the game -/
inductive Player : Type
| A | B | C | D

/-- The state of the game at any point -/
structure GameState :=
  (tokens : Player → ℕ)

/-- Initial state of the game -/
def initial_state : GameState :=
  { tokens := λ p => match p with
    | Player.A => 12
    | Player.B => 11
    | Player.C => 10
    | Player.D => 9 }

/-- Determines if the game has ended -/
def game_ended (state : GameState) : Prop :=
  ∃ p, state.tokens p = 0

/-- Simulates one round of the game -/
def play_round (state : GameState) : GameState :=
  sorry  -- Implementation details omitted

/-- The number of rounds played before the game ends -/
def rounds_played (state : GameState) : ℕ :=
  sorry  -- Implementation details omitted

/-- The main theorem stating that the game ends after exactly 3 rounds -/
theorem game_ends_in_three_rounds :
  rounds_played initial_state = 3 :=
sorry

end NUMINAMATH_CALUDE_game_ends_in_three_rounds_l3454_345423


namespace NUMINAMATH_CALUDE_xiao_ming_reading_problem_l3454_345491

/-- Represents the problem of finding the minimum number of pages to read per day -/
def min_pages_per_day (total_pages : ℕ) (total_days : ℕ) (initial_days : ℕ) (initial_pages_per_day : ℕ) : ℕ :=
  let remaining_days := total_days - initial_days
  let remaining_pages := total_pages - (initial_days * initial_pages_per_day)
  (remaining_pages + remaining_days - 1) / remaining_days

/-- Theorem stating the solution to Xiao Ming's reading problem -/
theorem xiao_ming_reading_problem :
  min_pages_per_day 72 10 2 5 = 8 :=
by sorry

end NUMINAMATH_CALUDE_xiao_ming_reading_problem_l3454_345491


namespace NUMINAMATH_CALUDE_min_value_of_f_l3454_345494

def f (x : ℝ) := x^2 - 2*x + 3

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3454_345494


namespace NUMINAMATH_CALUDE_company_handshakes_l3454_345482

/-- Represents the number of handshakes between employees of different departments -/
def handshakes (total_employees : ℕ) (dept_x_employees : ℕ) (dept_y_employees : ℕ) : ℕ :=
  dept_x_employees * dept_y_employees

/-- Theorem stating the number of handshakes between employees of different departments -/
theorem company_handshakes :
  ∃ (total_employees dept_x_employees dept_y_employees : ℕ),
    total_employees = 50 ∧
    dept_x_employees = 30 ∧
    dept_y_employees = 20 ∧
    total_employees = dept_x_employees + dept_y_employees ∧
    handshakes total_employees dept_x_employees dept_y_employees = 600 := by
  sorry

end NUMINAMATH_CALUDE_company_handshakes_l3454_345482


namespace NUMINAMATH_CALUDE_parabola_above_l3454_345413

theorem parabola_above (k : ℝ) : 
  (∀ x : ℝ, 2*x^2 - 2*k*x + (k^2 + 2*k + 2) > x^2 + 2*k*x - 2*k^2 - 1) ↔ 
  (-1 < k ∧ k < 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_above_l3454_345413


namespace NUMINAMATH_CALUDE_doubled_average_l3454_345458

theorem doubled_average (n : ℕ) (original_avg : ℚ) (h1 : n = 12) (h2 : original_avg = 50) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 100 := by sorry

end NUMINAMATH_CALUDE_doubled_average_l3454_345458


namespace NUMINAMATH_CALUDE_fish_sold_correct_l3454_345484

/-- The number of fish initially in stock -/
def initial_stock : ℕ := 200

/-- The number of fish in the new stock -/
def new_stock : ℕ := 200

/-- The final number of fish in stock -/
def final_stock : ℕ := 300

/-- The fraction of remaining fish that become spoiled -/
def spoilage_rate : ℚ := 1/3

/-- The number of fish sold -/
def fish_sold : ℕ := 50

theorem fish_sold_correct :
  (initial_stock - fish_sold - (initial_stock - fish_sold) * spoilage_rate + new_stock : ℚ) = final_stock :=
sorry

end NUMINAMATH_CALUDE_fish_sold_correct_l3454_345484


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3454_345459

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3454_345459


namespace NUMINAMATH_CALUDE_union_of_sets_l3454_345455

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {m : ℕ | m = 1 ∨ m = 4 ∨ m = 7}

theorem union_of_sets (h : A ∩ B = {1, 4}) : A ∪ B = {1, 2, 3, 4, 7} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3454_345455


namespace NUMINAMATH_CALUDE_unique_ecuadorian_number_l3454_345456

def is_ecuadorian (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- Three-digit number
  n % 10 ≠ 0 ∧  -- Does not end in 0
  n % 36 = 0 ∧  -- Multiple of 36
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) > 0 ∧  -- abc - cba > 0
  (n - (n % 10 * 100 + (n / 10 % 10) * 10 + n / 100)) % 36 = 0  -- (abc - cba) is multiple of 36

theorem unique_ecuadorian_number : ∃! n : ℕ, is_ecuadorian n ∧ n = 864 := by sorry

end NUMINAMATH_CALUDE_unique_ecuadorian_number_l3454_345456


namespace NUMINAMATH_CALUDE_triangleCount_is_sixteen_l3454_345469

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid :=
  (rows : ℕ)

/-- Counts the number of small triangles in a triangular grid -/
def countSmallTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows * (grid.rows + 1)) / 2

/-- Counts the number of medium triangles in a triangular grid -/
def countMediumTriangles (grid : TriangularGrid) : ℕ :=
  (grid.rows - 1) * grid.rows / 2

/-- Counts the number of large triangles in a triangular grid -/
def countLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the number of extra large triangles in a triangular grid -/
def countExtraLargeTriangles (grid : TriangularGrid) : ℕ := 1

/-- Counts the total number of triangles in a triangular grid -/
def countTotalTriangles (grid : TriangularGrid) : ℕ :=
  countSmallTriangles grid + countMediumTriangles grid + 
  countLargeTriangles grid + countExtraLargeTriangles grid

theorem triangleCount_is_sixteen :
  ∀ (grid : TriangularGrid), grid.rows = 4 → countTotalTriangles grid = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangleCount_is_sixteen_l3454_345469


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l3454_345425

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → (i⁻¹ : ℂ) + i^2015 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l3454_345425


namespace NUMINAMATH_CALUDE_system_solution_l3454_345427

theorem system_solution (x y z : ℝ) 
  (eq1 : x + y + z = 10)
  (eq2 : x * z = y^2)
  (eq3 : z^2 + y^2 = x^2) :
  z = 5 - Real.sqrt (Real.sqrt 3125 - 50) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3454_345427


namespace NUMINAMATH_CALUDE_subtract_equations_l3454_345465

theorem subtract_equations (x y : ℝ) :
  (4 * x - 3 * y = 2) ∧ (4 * x + y = 10) → 4 * y = 8 := by
  sorry

end NUMINAMATH_CALUDE_subtract_equations_l3454_345465


namespace NUMINAMATH_CALUDE_thabo_owns_280_books_l3454_345412

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabos_books : BookCollection where
  hardcover_nonfiction := 55
  paperback_nonfiction := 55 + 20
  paperback_fiction := 2 * (55 + 20)

/-- The total number of books in a collection -/
def total_books (bc : BookCollection) : ℕ :=
  bc.hardcover_nonfiction + bc.paperback_nonfiction + bc.paperback_fiction

/-- Theorem stating that Thabo owns 280 books in total -/
theorem thabo_owns_280_books : total_books thabos_books = 280 := by
  sorry

end NUMINAMATH_CALUDE_thabo_owns_280_books_l3454_345412


namespace NUMINAMATH_CALUDE_wand_price_l3454_345422

theorem wand_price (P : ℚ) : (P * (1/8) = 8) → P = 64 := by
  sorry

end NUMINAMATH_CALUDE_wand_price_l3454_345422


namespace NUMINAMATH_CALUDE_wooden_stick_problem_xiao_hong_age_problem_l3454_345435

-- Problem 1: Wooden stick
theorem wooden_stick_problem (x : ℝ) :
  60 - 2 * x = 10 → x = 25 := by sorry

-- Problem 2: Xiao Hong's age
theorem xiao_hong_age_problem (y : ℝ) :
  2 * y + 10 = 30 → y = 10 := by sorry

end NUMINAMATH_CALUDE_wooden_stick_problem_xiao_hong_age_problem_l3454_345435


namespace NUMINAMATH_CALUDE_son_work_time_l3454_345410

-- Define the work rates
def man_rate : ℚ := 1 / 6
def combined_rate : ℚ := 1 / 3

-- Define the son's work rate
def son_rate : ℚ := combined_rate - man_rate

-- Theorem to prove
theorem son_work_time : (1 : ℚ) / son_rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l3454_345410


namespace NUMINAMATH_CALUDE_geometric_progression_identity_l3454_345420

theorem geometric_progression_identity 
  (a r : ℝ) (n p k : ℕ) (A B C : ℝ) 
  (hA : A = a * r^(n - 1)) 
  (hB : B = a * r^(p - 1)) 
  (hC : C = a * r^(k - 1)) :
  A^(p - k) * B^(k - n) * C^(n - p) = 1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_progression_identity_l3454_345420


namespace NUMINAMATH_CALUDE_garage_visitors_l3454_345403

/-- Given a number of cars, selections per car, and selections per client,
    calculate the number of clients who visited the garage. -/
def clientsVisited (numCars : ℕ) (selectionsPerCar : ℕ) (selectionsPerClient : ℕ) : ℕ :=
  (numCars * selectionsPerCar) / selectionsPerClient

/-- Theorem stating that given 15 cars, where each car is selected exactly 3 times,
    and each client selects 3 cars, the number of clients who visited the garage is 15. -/
theorem garage_visitors :
  clientsVisited 15 3 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_garage_visitors_l3454_345403


namespace NUMINAMATH_CALUDE_fraction_inequality_l3454_345447

theorem fraction_inequality (x y z a b c r : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hr : r > 0) :
  (x + y + a + b) / (x + y + a + b + c + r) + (y + z + b + c) / (y + z + a + b + c + r) >
  (x + z + a + c) / (x + z + a + b + c + r) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3454_345447


namespace NUMINAMATH_CALUDE_tony_future_age_l3454_345430

def jacob_age : ℕ := 24
def tony_age : ℕ := jacob_age / 2
def years_passed : ℕ := 6

theorem tony_future_age :
  tony_age + years_passed = 18 := by
  sorry

end NUMINAMATH_CALUDE_tony_future_age_l3454_345430


namespace NUMINAMATH_CALUDE_intersection_condition_longest_chord_l3454_345431

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem 1: Intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem 2: Longest chord
theorem longest_chord :
  ∃ x y : ℝ, ellipse x y ∧ line x y 0 ∧
  ∀ m x' y' : ℝ, ellipse x' y' ∧ line x' y' m →
    (x - y)^2 ≥ (x' - y')^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_longest_chord_l3454_345431


namespace NUMINAMATH_CALUDE_jacks_paycheck_l3454_345497

theorem jacks_paycheck (paycheck : ℝ) : 
  (paycheck * 0.8 * 0.2 = 20) → paycheck = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_paycheck_l3454_345497


namespace NUMINAMATH_CALUDE_dance_team_members_l3454_345495

theorem dance_team_members :
  ∀ (track_members choir_members dance_members : ℕ),
    track_members + choir_members + dance_members = 100 →
    choir_members = 2 * track_members →
    dance_members = choir_members + 10 →
    dance_members = 46 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_members_l3454_345495


namespace NUMINAMATH_CALUDE_min_value_theorem_l3454_345433

theorem min_value_theorem (x y : ℝ) (h1 : x * y = 1/2) (h2 : 0 < x ∧ x < 1) (h3 : 0 < y ∧ y < 1) :
  (2 / (1 - x)) + (1 / (1 - y)) ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3454_345433


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3454_345401

theorem root_equation_implies_expression_value (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → (m-1)^2 - (m-3)*(m+3) - (m-1)*(m-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l3454_345401


namespace NUMINAMATH_CALUDE_angle_372_in_first_quadrant_l3454_345400

/-- An angle is in the first quadrant if it is between 0° and 90° (exclusive) when reduced to the range [0°, 360°) -/
def is_in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem: An angle of 372° is located in the first quadrant -/
theorem angle_372_in_first_quadrant :
  is_in_first_quadrant 372 := by
  sorry


end NUMINAMATH_CALUDE_angle_372_in_first_quadrant_l3454_345400


namespace NUMINAMATH_CALUDE_inequality_solution_l3454_345489

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  (x < 2 ∨ (3 < x ∧ x < 4) ∨ 5 < x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3454_345489


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3454_345448

theorem right_triangle_hypotenuse_and_perimeter :
  ∀ (a b c : ℝ),
  a = 60 →
  b = 80 →
  c^2 = a^2 + b^2 →
  c = 100 ∧ (a + b + c = 240) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_and_perimeter_l3454_345448


namespace NUMINAMATH_CALUDE_factorization_equality_l3454_345414

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 3 * y = 3 * y * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3454_345414


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3454_345461

theorem imaginary_part_of_z (z : ℂ) : z * (1 - 2*I) = Complex.abs (3 + 4*I) → Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3454_345461


namespace NUMINAMATH_CALUDE_train_station_problem_l3454_345462

theorem train_station_problem :
  ∀ (x v : ℕ),
  v > 3 →
  x = (2 * v) / (v - 3) →
  x - 5 > 0 →
  x / v - (x - 5) / 3 = 1 →
  (x = 8 ∧ v = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_train_station_problem_l3454_345462


namespace NUMINAMATH_CALUDE_part_one_part_two_l3454_345428

-- Define A and B as functions of a and b
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b

def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem for part (1)
theorem part_one (a b : ℝ) : 2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := by
  sorry

-- Theorem for part (2)
theorem part_two : (∀ a b : ℝ, ∃ c : ℝ, 2 * A a b - B a b = c) → (∀ a : ℝ, 2 * A a 2 - B a 2 = 2 * A 0 2 - B 0 2) := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3454_345428


namespace NUMINAMATH_CALUDE_solve_equation_l3454_345481

theorem solve_equation (x : ℝ) (h : 5 + 7 / x = 6 - 5 / x) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3454_345481


namespace NUMINAMATH_CALUDE_square_rotation_overlap_area_l3454_345450

theorem square_rotation_overlap_area (β : Real) (h1 : 0 < β) (h2 : β < π/2) (h3 : Real.sin β = 3/5) :
  let side_length : Real := 2
  let overlap_area := 2 * (1/2 * side_length * (side_length * ((1 - Real.tan (β/2)) / (1 + Real.tan (β/2)))))
  overlap_area = 2 := by
sorry

end NUMINAMATH_CALUDE_square_rotation_overlap_area_l3454_345450


namespace NUMINAMATH_CALUDE_iggy_thursday_miles_l3454_345488

/-- Represents Iggy's running schedule for a week --/
structure RunningSchedule where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total miles run in a week --/
def totalMiles (schedule : RunningSchedule) : Nat :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Calculates the total minutes run in a week given a pace in minutes per mile --/
def totalMinutes (schedule : RunningSchedule) (pace : Nat) : Nat :=
  (totalMiles schedule) * pace

/-- Theorem stating that Iggy ran 8 miles on Thursday --/
theorem iggy_thursday_miles :
  ∀ (schedule : RunningSchedule) (pace : Nat),
    schedule.monday = 3 →
    schedule.tuesday = 4 →
    schedule.wednesday = 6 →
    schedule.friday = 3 →
    pace = 10 →
    totalMinutes schedule pace = 4 * 60 →
    schedule.thursday = 8 := by
  sorry


end NUMINAMATH_CALUDE_iggy_thursday_miles_l3454_345488


namespace NUMINAMATH_CALUDE_lemonade_stand_solution_l3454_345493

/-- Represents the lemonade stand problem --/
def lemonade_stand_problem (G : ℚ) : Prop :=
  let glasses_per_gallon : ℚ := 16
  let cost_per_gallon : ℚ := 3.5
  let price_per_glass : ℚ := 1
  let glasses_drunk : ℚ := 5
  let glasses_unsold : ℚ := 6
  let net_profit : ℚ := 14
  let total_glasses := G * glasses_per_gallon
  let glasses_sold := total_glasses - glasses_drunk - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := G * cost_per_gallon
  revenue - cost = net_profit

/-- The solution to the lemonade stand problem --/
theorem lemonade_stand_solution :
  ∃ G : ℚ, lemonade_stand_problem G ∧ G = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_solution_l3454_345493


namespace NUMINAMATH_CALUDE_physics_marks_l3454_345473

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 80)
  (avg_PM : (P + M) / 2 = 90)
  (avg_PC : (P + C) / 2 = 70) :
  P = 80 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l3454_345473


namespace NUMINAMATH_CALUDE_vector_subtraction_l3454_345411

/-- Given two vectors OA and OB in 2D space, prove that the vector AB is their difference -/
theorem vector_subtraction (OA OB : ℝ × ℝ) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  OB - OA = (-9, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l3454_345411


namespace NUMINAMATH_CALUDE_log_stack_sum_l3454_345443

theorem log_stack_sum (n : ℕ) (a l : ℕ) (h1 : n = 12) (h2 : a = 15) (h3 : l = 4) :
  n * (a + l) / 2 = 114 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l3454_345443


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3454_345434

theorem inequality_equivalence (x : ℝ) : (2 * x - 4 ≥ 0) ↔ (x ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3454_345434


namespace NUMINAMATH_CALUDE_max_trailing_zeros_consecutive_two_digit_numbers_l3454_345418

/-- Two-digit number type -/
def TwoDigitNumber := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

/-- Function to count trailing zeros of a natural number -/
def countTrailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The maximum number of consecutive zeros at the end of the product 
    of two consecutive two-digit numbers is 2 -/
theorem max_trailing_zeros_consecutive_two_digit_numbers : 
  ∃ (a : TwoDigitNumber), 
    let b : TwoDigitNumber := ⟨a.val + 1, sorry⟩
    countTrailingZeros (a.val * b.val) = 2 ∧ 
    ∀ (x : TwoDigitNumber), 
      let y : TwoDigitNumber := ⟨x.val + 1, sorry⟩
      countTrailingZeros (x.val * y.val) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_consecutive_two_digit_numbers_l3454_345418


namespace NUMINAMATH_CALUDE_absolute_value_half_l3454_345477

theorem absolute_value_half (a : ℝ) : 
  |a| = 1/2 → (a = 1/2 ∨ a = -1/2) := by sorry

end NUMINAMATH_CALUDE_absolute_value_half_l3454_345477


namespace NUMINAMATH_CALUDE_g_of_3_equals_101_l3454_345499

-- Define the polynomial g(x)
def g (x : ℝ) : ℝ := 5 * x^3 - 4 * x^2 + 3 * x - 7

-- Theorem stating that g(3) = 101
theorem g_of_3_equals_101 : g 3 = 101 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_101_l3454_345499


namespace NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l3454_345407

theorem percentage_of_women_in_non_union (total : ℝ) (h1 : total > 0) : 
  let men := 0.48 * total
  let unionized := 0.60 * total
  let non_unionized := total - unionized
  let women_non_union := 0.85 * non_unionized
  women_non_union / non_unionized = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_non_union_l3454_345407


namespace NUMINAMATH_CALUDE_painting_job_completion_time_l3454_345492

/-- Represents the painting job with given conditions -/
structure PaintingJob where
  original_men : ℕ
  original_days : ℕ
  additional_men : ℕ
  efficiency_increase : ℚ

/-- Calculates the number of days required to complete the job with additional skilled workers -/
def days_with_skilled_workers (job : PaintingJob) : ℚ :=
  let total_man_days := job.original_men * job.original_days
  let original_daily_output := job.original_men
  let skilled_daily_output := job.additional_men * (1 + job.efficiency_increase)
  let total_daily_output := original_daily_output + skilled_daily_output
  total_man_days / total_daily_output

/-- The main theorem stating that the job will be completed in 4 days -/
theorem painting_job_completion_time :
  let job := PaintingJob.mk 10 6 4 (1/4)
  days_with_skilled_workers job = 4 := by
  sorry

#eval days_with_skilled_workers (PaintingJob.mk 10 6 4 (1/4))

end NUMINAMATH_CALUDE_painting_job_completion_time_l3454_345492


namespace NUMINAMATH_CALUDE_speedster_fraction_l3454_345466

/-- Represents the inventory of vehicles -/
structure Inventory where
  speedsters : ℕ
  nonSpeedsters : ℕ

/-- The fraction of Speedsters that are convertibles -/
def convertibleFraction : ℚ := 3/5

/-- The number of Speedster convertibles -/
def speedsterConvertibles : ℕ := 54

/-- The number of non-Speedster vehicles -/
def nonSpeedsterCount : ℕ := 30

/-- Theorem: The fraction of Speedsters in the inventory is 3/4 -/
theorem speedster_fraction (inv : Inventory) 
  (h1 : inv.speedsters * convertibleFraction = speedsterConvertibles)
  (h2 : inv.nonSpeedsters = nonSpeedsterCount) :
  (inv.speedsters : ℚ) / (inv.speedsters + inv.nonSpeedsters) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_speedster_fraction_l3454_345466
