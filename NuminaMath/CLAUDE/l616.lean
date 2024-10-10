import Mathlib

namespace power_sum_inequality_l616_61679

theorem power_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^4 + b^4 + c^4 ≥ a*b*c*(a + b + c) := by
  sorry

end power_sum_inequality_l616_61679


namespace boys_distribution_l616_61638

theorem boys_distribution (total_amount : ℕ) (additional_amount : ℕ) : 
  total_amount = 5040 →
  additional_amount = 80 →
  ∃ (x : ℕ), 
    x * (total_amount / 18 + additional_amount) = total_amount ∧
    x = 14 := by
  sorry

end boys_distribution_l616_61638


namespace larger_root_of_quadratic_l616_61694

theorem larger_root_of_quadratic (x : ℝ) : 
  x^2 + 17*x - 72 = 0 → x ≤ 3 :=
by sorry

end larger_root_of_quadratic_l616_61694


namespace perpendicular_iff_x_eq_neg_three_l616_61673

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_iff_x_eq_neg_three :
  ∀ x : ℝ, perpendicular (x, -3) (2, -2) ↔ x = -3 := by
  sorry

end perpendicular_iff_x_eq_neg_three_l616_61673


namespace systematic_sampling_l616_61623

theorem systematic_sampling 
  (total_employees : Nat) 
  (sample_size : Nat) 
  (fifth_sample : Nat) :
  total_employees = 200 →
  sample_size = 40 →
  fifth_sample = 23 →
  ∃ (start : Nat), 
    start + 4 * (total_employees / sample_size) = fifth_sample ∧
    start + 7 * (total_employees / sample_size) = 38 :=
by sorry

end systematic_sampling_l616_61623


namespace three_digit_to_four_digit_l616_61680

theorem three_digit_to_four_digit (a : ℕ) (h : 100 ≤ a ∧ a ≤ 999) :
  (10 * a + 1 : ℕ) = 1000 + (a - 100) * 10 + 1 :=
by sorry

end three_digit_to_four_digit_l616_61680


namespace tile_border_ratio_l616_61620

theorem tile_border_ratio (s d : ℝ) (h_positive : s > 0 ∧ d > 0) : 
  (15 * s)^2 / ((15 * s + 2 * 15 * d)^2) = 3/4 → d/s = 1/13 := by
  sorry

end tile_border_ratio_l616_61620


namespace opposite_direction_speed_l616_61666

/-- Given two people moving in opposite directions, this theorem proves the speed of one person
    given the speed of the other and their final distance after a certain time. -/
theorem opposite_direction_speed
  (time : ℝ)
  (speed_person2 : ℝ)
  (final_distance : ℝ)
  (h1 : time > 0)
  (h2 : speed_person2 > 0)
  (h3 : final_distance > 0)
  (h4 : final_distance = (speed_person1 + speed_person2) * time)
  (h5 : time = 4)
  (h6 : speed_person2 = 3)
  (h7 : final_distance = 36) :
  speed_person1 = 6 :=
sorry

end opposite_direction_speed_l616_61666


namespace ratio_of_sum_to_difference_l616_61678

theorem ratio_of_sum_to_difference (a b : ℝ) : 
  a > 0 → b > 0 → a > b → a + b = 5 * (a - b) → a / b = 3 / 2 := by
  sorry

end ratio_of_sum_to_difference_l616_61678


namespace percentage_difference_l616_61688

theorem percentage_difference (A B C x : ℝ) : 
  C > B → B > A → A > 0 → C = A + 2*B → A = B * (1 - x/100) → 
  x = 100 * ((B - A) / B) := by
sorry

end percentage_difference_l616_61688


namespace max_value_of_function_l616_61689

theorem max_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  ∃ (y : ℝ), y = x^2 * (1 - 2*x) ∧ y ≤ 1/27 ∧ ∃ (x0 : ℝ), 0 < x0 ∧ x0 < 1/2 ∧ x0^2 * (1 - 2*x0) = 1/27 :=
sorry

end max_value_of_function_l616_61689


namespace double_discount_price_l616_61602

/-- Proves that if a price P is discounted twice by 25% and the final price is $15, then the original price P is equal to $26.67 -/
theorem double_discount_price (P : ℝ) : 
  (0.75 * (0.75 * P) = 15) → P = 26.67 := by
sorry

end double_discount_price_l616_61602


namespace inequality_proof_l616_61640

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) := by
sorry

end inequality_proof_l616_61640


namespace store_earnings_is_400_l616_61656

/-- Calculates the total earnings of a clothing store selling shirts and jeans -/
def store_earnings (num_shirts : ℕ) (num_jeans : ℕ) (shirt_price : ℕ) : ℕ :=
  let jeans_price := 2 * shirt_price
  num_shirts * shirt_price + num_jeans * jeans_price

/-- Theorem: The clothing store will earn $400 if all shirts and jeans are sold -/
theorem store_earnings_is_400 :
  store_earnings 20 10 10 = 400 := by
sorry

end store_earnings_is_400_l616_61656


namespace dishonest_dealer_profit_percentage_l616_61658

/-- A dishonest dealer's profit percentage when using underweight measurements --/
theorem dishonest_dealer_profit_percentage 
  (actual_weight : ℝ) 
  (claimed_weight : ℝ) 
  (actual_weight_positive : 0 < actual_weight)
  (claimed_weight_greater : actual_weight < claimed_weight) : 
  (claimed_weight - actual_weight) / actual_weight * 100 = 
  (1000 - 920) / 920 * 100 := by
sorry

#eval (1000 - 920) / 920 * 100  -- To show the approximate result

end dishonest_dealer_profit_percentage_l616_61658


namespace horner_rule_v₄_l616_61663

-- Define the polynomial coefficients
def a₀ : ℤ := 12
def a₁ : ℤ := 35
def a₂ : ℤ := -8
def a₃ : ℤ := 6
def a₄ : ℤ := 5
def a₅ : ℤ := 3

-- Define x
def x : ℤ := -2

-- Define Horner's Rule steps
def v₀ : ℤ := a₅
def v₁ : ℤ := v₀ * x + a₄
def v₂ : ℤ := v₁ * x + a₃
def v₃ : ℤ := v₂ * x + a₂
def v₄ : ℤ := v₃ * x + a₁

-- Theorem statement
theorem horner_rule_v₄ : v₄ = 83 := by
  sorry

end horner_rule_v₄_l616_61663


namespace function_value_proof_l616_61652

theorem function_value_proof (f : ℝ → ℝ) :
  (3 : ℝ) + 17 = 60 * f 3 → f 3 = 1/3 := by
  sorry

end function_value_proof_l616_61652


namespace parallel_vectors_x_value_l616_61677

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 :=
by
  sorry


end parallel_vectors_x_value_l616_61677


namespace point_d_from_c_l616_61669

/-- Given two points C and D in the Cartesian coordinate system, prove that D is obtained from C by moving 3 units downwards -/
theorem point_d_from_c (C D : ℝ × ℝ) : 
  C = (1, 2) → D = (1, -1) → 
  (C.2 - D.2 = 3) ∧ (D.2 < C.2) := by
  sorry

end point_d_from_c_l616_61669


namespace peanut_butter_probability_l616_61665

def jenny_peanut_butter : ℕ := 40
def jenny_chocolate_chip : ℕ := 50
def marcus_peanut_butter : ℕ := 30
def marcus_lemon : ℕ := 20

def total_cookies : ℕ := jenny_peanut_butter + jenny_chocolate_chip + marcus_peanut_butter + marcus_lemon
def peanut_butter_cookies : ℕ := jenny_peanut_butter + marcus_peanut_butter

theorem peanut_butter_probability :
  (peanut_butter_cookies : ℚ) / (total_cookies : ℚ) = 1/2 := by
  sorry

end peanut_butter_probability_l616_61665


namespace smallest_integer_with_remainder_one_l616_61636

theorem smallest_integer_with_remainder_one : ∃! k : ℕ,
  k > 1 ∧
  k % 19 = 1 ∧
  k % 7 = 1 ∧
  k % 3 = 1 ∧
  ∀ m : ℕ, m > 1 ∧ m % 19 = 1 ∧ m % 7 = 1 ∧ m % 3 = 1 → k ≤ m :=
by
  use 400
  sorry

end smallest_integer_with_remainder_one_l616_61636


namespace ellipse_semi_major_axis_l616_61616

theorem ellipse_semi_major_axis (m : ℝ) : 
  (∀ x y : ℝ, x^2 / m + y^2 / 4 = 1) →  -- Ellipse equation
  (m > 4) →                            -- Semi-major axis > Semi-minor axis
  (m = (2 : ℝ)^2 + 4) →                -- Relationship between a^2, b^2, and c^2
  (m = 5) :=
by sorry

end ellipse_semi_major_axis_l616_61616


namespace crayons_per_day_l616_61683

theorem crayons_per_day (total_crayons : ℕ) (crayons_per_box : ℕ) 
  (h1 : total_crayons = 321)
  (h2 : crayons_per_box = 7) : 
  (total_crayons / crayons_per_box : ℕ) = 45 := by
  sorry

end crayons_per_day_l616_61683


namespace three_kopeck_count_l616_61615

/-- Represents the denomination of a coin -/
inductive Denomination
| One
| Two
| Three

/-- Represents a row of coins -/
def CoinRow := List Denomination

/-- Checks if there's at least one coin between any two one-kopeck coins -/
def validOneKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least two coins between any two two-kopeck coins -/
def validTwoKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Checks if there's at least three coins between any two three-kopeck coins -/
def validThreeKopeckSpacing (row : CoinRow) : Prop := sorry

/-- Counts the number of three-kopeck coins in the row -/
def countThreeKopecks (row : CoinRow) : Nat := sorry

theorem three_kopeck_count (row : CoinRow) :
  row.length = 101 →
  validOneKopeckSpacing row →
  validTwoKopeckSpacing row →
  validThreeKopeckSpacing row →
  (countThreeKopecks row = 25 ∨ countThreeKopecks row = 26) :=
by sorry

end three_kopeck_count_l616_61615


namespace angle_sum_in_circle_l616_61693

theorem angle_sum_in_circle (x : ℝ) : 
  (3 * x + 7 * x + 4 * x + 2 * x + x = 360) → x = 360 / 17 := by
  sorry

end angle_sum_in_circle_l616_61693


namespace equal_angle_vector_l616_61642

theorem equal_angle_vector (a b c : ℝ × ℝ) : 
  a = (1, 2) → 
  b = (4, 2) → 
  c ≠ (0, 0) → 
  (c.1 * a.1 + c.2 * a.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (a.1^2 + a.2^2)) = 
  (c.1 * b.1 + c.2 * b.2) / (Real.sqrt (c.1^2 + c.2^2) * Real.sqrt (b.1^2 + b.2^2)) → 
  ∃ (k : ℝ), k ≠ 0 ∧ c = (k, k) := by
sorry

end equal_angle_vector_l616_61642


namespace log_division_simplification_l616_61633

theorem log_division_simplification : 
  Real.log 16 / Real.log (1 / 16) = -1 := by sorry

end log_division_simplification_l616_61633


namespace max_subjects_per_teacher_l616_61657

theorem max_subjects_per_teacher 
  (total_subjects : Nat) 
  (min_teachers : Nat) 
  (h1 : total_subjects = 18) 
  (h2 : min_teachers = 6) : 
  Nat.ceil (total_subjects / min_teachers) = 3 := by
  sorry

end max_subjects_per_teacher_l616_61657


namespace rectangle_problem_l616_61698

theorem rectangle_problem (A B C D E F G H I : ℕ) : 
  (A * B = D * E) →  -- Areas of ABCD and DEFG are equal
  (A * B = C * H) →  -- Areas of ABCD and CEIH are equal
  (B = 43) →         -- BC = 43
  (D > E) →          -- Assume DG > DE
  (D = 1892) →       -- DG = 1892
  True               -- Conclusion (to be proved)
  := by sorry

end rectangle_problem_l616_61698


namespace product_of_digits_is_64_l616_61671

/-- Represents a number in different bases -/
structure NumberInBases where
  base10 : ℕ
  b : ℕ
  base_b : ℕ
  base_b_plus_2 : ℕ

/-- Calculates the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating the product of digits of N is 64 -/
theorem product_of_digits_is_64 (N : NumberInBases) 
  (h1 : N.base_b = 503)
  (h2 : N.base_b_plus_2 = 305)
  (h3 : N.b > 0) : 
  productOfDigits N.base10 = 64 := by sorry

end product_of_digits_is_64_l616_61671


namespace binomial_coefficient_39_5_l616_61605

theorem binomial_coefficient_39_5 : 
  let n : ℕ := 39
  let binomial := n * (n - 1) * (n - 2) * (n - 3) * (n - 4) / (2 * 3 * 4 * 5)
  binomial = 575757 := by sorry

end binomial_coefficient_39_5_l616_61605


namespace female_democrats_count_l616_61625

theorem female_democrats_count (total_participants : ℕ) 
  (female_participants male_participants : ℕ) 
  (female_democrats male_democrats : ℕ) : 
  total_participants = 840 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats + male_democrats = total_participants / 3 →
  female_democrats = 140 := by
  sorry

end female_democrats_count_l616_61625


namespace valid_a_equals_solution_set_l616_61631

/-- R(k) is the remainder when k is divided by p -/
def R (p k : ℕ) : ℕ := k % p

/-- The set of valid a values -/
def valid_a_set (p : ℕ) : Set ℕ :=
  {a | a > 0 ∧ ∀ m ∈ Finset.range (p - 1), m + 1 + R p (m * a) > a}

/-- The set of solutions described in the problem -/
def solution_set (p : ℕ) : Set ℕ :=
  {p - 1} ∪ {a | ∃ s, 1 ≤ s ∧ s ≤ p - 1 ∧ a = (p - 1) / s}

theorem valid_a_equals_solution_set (p : ℕ) (hp : p.Prime ∧ p ≥ 5) :
  valid_a_set p = solution_set p := by
  sorry

end valid_a_equals_solution_set_l616_61631


namespace quadratic_trinomial_constant_l616_61660

/-- Given that x^{|m|}+(m-2)x-10 is a quadratic trinomial where m is a constant, prove that m = -2 -/
theorem quadratic_trinomial_constant (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, x^(|m|) + (m-2)*x - 10 = a*x^2 + b*x + c ∧ a ≠ 0 ∧ b ≠ 0) → 
  m = -2 := by
sorry

end quadratic_trinomial_constant_l616_61660


namespace correct_distinct_arrangements_l616_61687

/-- The number of distinct arrangements to distribute 5 students into two dormitories,
    with each dormitory accommodating at least 2 students. -/
def distinct_arrangements : ℕ := 20

/-- The total number of students to be distributed. -/
def total_students : ℕ := 5

/-- The number of dormitories. -/
def num_dormitories : ℕ := 2

/-- The minimum number of students that must be in each dormitory. -/
def min_students_per_dormitory : ℕ := 2

/-- Theorem stating that the number of distinct arrangements is correct. -/
theorem correct_distinct_arrangements :
  distinct_arrangements = 20 ∧
  total_students = 5 ∧
  num_dormitories = 2 ∧
  min_students_per_dormitory = 2 :=
by sorry

end correct_distinct_arrangements_l616_61687


namespace geometric_sequence_sum_l616_61609

/-- A geometric sequence with common ratio q > 1 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 1 + a 6 = 33 →
  a 2 * a 5 = 32 →
  a 3 + a 8 = 132 := by
sorry

end geometric_sequence_sum_l616_61609


namespace negation_equivalence_l616_61607

theorem negation_equivalence (x : ℝ) :
  ¬(x^2 - x + 3 > 0) ↔ (x^2 - x + 3 ≤ 0) :=
by sorry

end negation_equivalence_l616_61607


namespace sum_distribution_l616_61621

/-- The sum distribution problem -/
theorem sum_distribution (p q r s t : ℝ) : 
  (q = 0.75 * p) →  -- q gets 75 cents for each dollar p gets
  (r = 0.50 * p) →  -- r gets 50 cents for each dollar p gets
  (s = 0.25 * p) →  -- s gets 25 cents for each dollar p gets
  (t = 0.10 * p) →  -- t gets 10 cents for each dollar p gets
  (s = 25) →        -- The share of s is 25 dollars
  (p + q + r + s + t = 260) := by  -- The total sum is 260 dollars
sorry


end sum_distribution_l616_61621


namespace ellipse_min_area_l616_61664

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_min_area (a b : ℝ) (h_ellipse : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
  ((x - 2)^2 + y^2 ≥ 4 ∧ (x + 2)^2 + y^2 ≥ 4)) :
  π * a * b ≥ (3 * Real.sqrt 3 / 2) * π := by
  sorry

#check ellipse_min_area

end ellipse_min_area_l616_61664


namespace product_remainder_by_10_l616_61651

theorem product_remainder_by_10 : (4219 * 2675 * 394082 * 5001) % 10 = 0 := by
  sorry

end product_remainder_by_10_l616_61651


namespace max_t_value_l616_61601

theorem max_t_value (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 10 →
  k < m → m < r → r < s → s < t →
  r ≤ 13 →
  t ≤ 20 := by
  sorry

end max_t_value_l616_61601


namespace original_average_l616_61626

theorem original_average (n : ℕ) (A : ℝ) (h1 : n = 7) (h2 : (5 * n * A) / n = 100) : A = 20 := by
  sorry

end original_average_l616_61626


namespace quadratic_root_value_l616_61644

theorem quadratic_root_value (x : ℝ) : x = -4 → Real.sqrt (1 - 2*x) = 3 := by
  sorry

end quadratic_root_value_l616_61644


namespace equal_numbers_l616_61696

theorem equal_numbers (a : Fin 100 → ℝ)
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
by sorry


end equal_numbers_l616_61696


namespace audrey_sleep_theorem_l616_61627

theorem audrey_sleep_theorem (total_sleep : ℝ) (dream_ratio : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : dream_ratio = 2/5) : 
  total_sleep - (dream_ratio * total_sleep) = 6 := by
  sorry

end audrey_sleep_theorem_l616_61627


namespace quadratic_two_distinct_roots_l616_61654

theorem quadratic_two_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 4*x₁ - m = 0 ∧ x₂^2 - 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end quadratic_two_distinct_roots_l616_61654


namespace floor_sqrt_95_l616_61632

theorem floor_sqrt_95 : ⌊Real.sqrt 95⌋ = 9 := by sorry

end floor_sqrt_95_l616_61632


namespace obtuse_triangle_area_bound_l616_61606

theorem obtuse_triangle_area_bound (a b c : ℝ) (h_obtuse : 0 < a ∧ 0 < b ∧ 0 < c ∧ c^2 > a^2 + b^2) 
  (h_longest : c = 4) (h_shortest : a = 2) : 
  (1/2 * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) ≤ 2 * Real.sqrt 3 := by
  sorry

end obtuse_triangle_area_bound_l616_61606


namespace sock_pair_count_l616_61668

/-- The number of ways to choose a pair of socks of different colors -/
def differentColorPairs (white brown blue : ℕ) : ℕ :=
  white * brown + brown * blue + white * blue

/-- Theorem: There are 47 ways to choose a pair of socks of different colors
    from 5 white socks, 4 brown socks, and 3 blue socks -/
theorem sock_pair_count :
  differentColorPairs 5 4 3 = 47 := by
  sorry

end sock_pair_count_l616_61668


namespace shaded_area_ratio_l616_61662

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ s : ℝ, s > 0 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = s^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = s^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = s^2

-- Define midpoint
def Midpoint (M X Y : ℝ × ℝ) : Prop :=
  M.1 = (X.1 + Y.1) / 2 ∧ M.2 = (X.2 + Y.2) / 2

-- Define the theorem
theorem shaded_area_ratio 
  (A B C D E F G H : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint D A B) 
  (h3 : Midpoint E B C) 
  (h4 : Midpoint F C A) 
  (h5 : Midpoint G D F) 
  (h6 : Midpoint H F E) :
  let shaded_area := 
    -- Area of triangle DEF + Area of three trapezoids
    (Real.sqrt 3 / 16 + 9 * Real.sqrt 3 / 32) * s^2
  let non_shaded_area := 
    -- Total area of triangle ABC - Shaded area
    (Real.sqrt 3 / 4 - 11 * Real.sqrt 3 / 32) * s^2
  shaded_area / non_shaded_area = 11 / 21 :=
by sorry

end shaded_area_ratio_l616_61662


namespace line_tangent_to_circle_l616_61684

/-- A circle with center O and radius 3 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A line m containing point P -/
structure Line :=
  (m : Set (ℝ × ℝ))
  (P : ℝ × ℝ)
  (h_P_on_m : P ∈ m)

/-- The distance between two points in ℝ² -/
def distance (A B : ℝ × ℝ) : ℝ := sorry

/-- Defines what it means for a line to be tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  ∃ (Q : ℝ × ℝ), Q ∈ l.m ∧ distance Q c.O = c.radius ∧
  ∀ (R : ℝ × ℝ), R ∈ l.m → R ≠ Q → distance R c.O > c.radius

theorem line_tangent_to_circle (c : Circle) (l : Line) :
  distance l.P c.O = c.radius →
  is_tangent l c :=
sorry

end line_tangent_to_circle_l616_61684


namespace fractional_equation_solutions_l616_61659

/-- The fractional equation in terms of x and m -/
def fractional_equation (x m : ℝ) : Prop :=
  3 * x / (x - 1) = m / (x - 1) + 2

theorem fractional_equation_solutions :
  (∃! x : ℝ, fractional_equation x 4) ∧
  (∀ x : ℝ, ¬fractional_equation x 3) ∧
  (∀ m : ℝ, m ≠ 3 → ∃ x : ℝ, fractional_equation x m) :=
sorry

end fractional_equation_solutions_l616_61659


namespace arc_length_30_degrees_l616_61682

/-- The length of an arc in a circle with radius 3 and central angle 30° is π/2 -/
theorem arc_length_30_degrees (r : ℝ) (θ : ℝ) (L : ℝ) : 
  r = 3 → θ = 30 * π / 180 → L = r * θ → L = π / 2 := by
  sorry

end arc_length_30_degrees_l616_61682


namespace cosine_values_for_special_angle_l616_61622

theorem cosine_values_for_special_angle (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π/3)) 
  (h2 : Real.sqrt 6 * Real.sin α + Real.sqrt 2 * Real.cos α = Real.sqrt 3) : 
  (Real.cos (α + π/6) = Real.sqrt 10 / 4) ∧ 
  (Real.cos (2*α + π/12) = (Real.sqrt 30 + Real.sqrt 2) / 8) := by
  sorry

end cosine_values_for_special_angle_l616_61622


namespace chris_least_money_l616_61634

-- Define the set of people
inductive Person : Type
  | Alice : Person
  | Bob : Person
  | Chris : Person
  | Dana : Person
  | Eve : Person

-- Define the money function
variable (money : Person → ℝ)

-- State the conditions
axiom different_amounts : ∀ p q : Person, p ≠ q → money p ≠ money q
axiom chris_less_than_bob : money Person.Chris < money Person.Bob
axiom dana_less_than_bob : money Person.Dana < money Person.Bob
axiom alice_more_than_chris : money Person.Chris < money Person.Alice
axiom eve_more_than_chris : money Person.Chris < money Person.Eve
axiom dana_equal_eve : money Person.Dana = money Person.Eve
axiom dana_less_than_alice : money Person.Dana < money Person.Alice
axiom bob_more_than_eve : money Person.Eve < money Person.Bob

-- State the theorem
theorem chris_least_money :
  ∀ p : Person, p ≠ Person.Chris → money Person.Chris ≤ money p :=
sorry

end chris_least_money_l616_61634


namespace ending_number_proof_l616_61635

theorem ending_number_proof (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k) →  -- n is divisible by 7
  n ≥ 21 →               -- n is at least 21 (first number after 18 divisible by 7)
  (21 + n) / 2 = 77/2 →  -- average of arithmetic sequence is 38.5
  n = 56 := by
sorry

end ending_number_proof_l616_61635


namespace product_of_repeating_decimals_l616_61672

/-- The decimal representation of 0.0808... -/
def repeating_decimal_08 : ℚ := 8 / 99

/-- The decimal representation of 0.3636... -/
def repeating_decimal_36 : ℚ := 36 / 99

/-- The product of 0.0808... and 0.3636... is equal to 288/9801 -/
theorem product_of_repeating_decimals : 
  repeating_decimal_08 * repeating_decimal_36 = 288 / 9801 := by
  sorry

end product_of_repeating_decimals_l616_61672


namespace sufficient_not_necessary_condition_necessary_not_sufficient_condition_l616_61686

-- Proposition A
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
sorry

-- Proposition D
theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b, a ≠ 0 ∧ a * b = 0) :=
sorry

end sufficient_not_necessary_condition_necessary_not_sufficient_condition_l616_61686


namespace negation_of_universal_quadrilateral_circumcircle_l616_61612

-- Define the type for quadrilaterals
variable (Quadrilateral : Type)

-- Define the property of having a circumcircle
variable (has_circumcircle : Quadrilateral → Prop)

-- Theorem stating the negation of "Every quadrilateral has a circumcircle"
-- is equivalent to "Some quadrilaterals do not have a circumcircle"
theorem negation_of_universal_quadrilateral_circumcircle :
  ¬(∀ q : Quadrilateral, has_circumcircle q) ↔ ∃ q : Quadrilateral, ¬(has_circumcircle q) :=
by sorry

end negation_of_universal_quadrilateral_circumcircle_l616_61612


namespace amusement_park_cost_per_trip_l616_61643

/-- The cost per trip to an amusement park given the following conditions:
  * Two season passes are purchased
  * Each pass costs 100 (in some currency unit)
  * One person uses their pass 35 times
  * Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (num_passes : ℕ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) 
  (h1 : pass_cost = 100) 
  (h2 : num_passes = 2) 
  (h3 : trips_person1 = 35) 
  (h4 : trips_person2 = 15) : 
  (num_passes * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end amusement_park_cost_per_trip_l616_61643


namespace min_reciprocal_sum_product_upper_bound_l616_61618

-- Define the set of positive real numbers
def PositiveReals := {x : ℝ | x > 0}

-- Define the condition x^2 + y^2 = x + y
def SumSquaresEqualSum (x y : ℝ) : Prop := x^2 + y^2 = x + y

-- Theorem 1: Minimum value of 1/x + 1/y is 2
theorem min_reciprocal_sum (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  1/x + 1/y ≥ 2 ∧ ∃ x y, x ∈ PositiveReals ∧ y ∈ PositiveReals ∧ SumSquaresEqualSum x y ∧ 1/x + 1/y = 2 :=
sorry

-- Theorem 2: (x+1)(y+1) < 5 for all x, y satisfying the conditions
theorem product_upper_bound (x y : ℝ) (hx : x ∈ PositiveReals) (hy : y ∈ PositiveReals)
  (h : SumSquaresEqualSum x y) :
  (x + 1) * (y + 1) < 5 :=
sorry

end min_reciprocal_sum_product_upper_bound_l616_61618


namespace complex_fraction_simplification_l616_61637

theorem complex_fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 7 / 15) = 15 / 13 := by
  sorry

end complex_fraction_simplification_l616_61637


namespace line_through_points_l616_61675

/-- A line with slope 4 passing through points (3,5), (a,7), and (-1,b) has a = 7/2 and b = -11 -/
theorem line_through_points (a b : ℚ) : 
  (((7 - 5) / (a - 3) = 4) ∧ ((b - 5) / (-1 - 3) = 4)) → 
  (a = 7/2 ∧ b = -11) := by
sorry

end line_through_points_l616_61675


namespace vector_sum_equals_one_five_l616_61614

/-- Given vectors a and b in R², prove that their sum is (1, 5) -/
theorem vector_sum_equals_one_five :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![-1, 2]
  (a + b) = ![1, 5] := by
  sorry

end vector_sum_equals_one_five_l616_61614


namespace stormi_lawn_mowing_charge_l616_61648

/-- Proves that Stormi charges $13 for mowing each lawn given the problem conditions -/
theorem stormi_lawn_mowing_charge : 
  (car_wash_count : ℕ) →
  (car_wash_price : ℚ) →
  (lawn_count : ℕ) →
  (bicycle_price : ℚ) →
  (additional_money_needed : ℚ) →
  car_wash_count = 3 →
  car_wash_price = 10 →
  lawn_count = 2 →
  bicycle_price = 80 →
  additional_money_needed = 24 →
  (bicycle_price - additional_money_needed - car_wash_count * car_wash_price) / lawn_count = 13 :=
by
  sorry

end stormi_lawn_mowing_charge_l616_61648


namespace lucky_larry_problem_l616_61661

theorem lucky_larry_problem (p q r s t : ℤ) 
  (hp : p = 2) (hq : q = 4) (hr : r = 6) (hs : s = 8) :
  p - (q - (r - (s - t))) = p - q - r + s - t → t = 2 := by
  sorry

end lucky_larry_problem_l616_61661


namespace sum_of_cubic_difference_l616_61613

theorem sum_of_cubic_difference (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 294 →
  a + b + c = 8 := by
  sorry

end sum_of_cubic_difference_l616_61613


namespace ratio_transformation_l616_61699

theorem ratio_transformation (a b c d x : ℚ) : 
  a = 4 ∧ b = 15 ∧ c = 3 ∧ d = 4 ∧ x = 29 →
  (a + x) / (b + x) = c / d := by
sorry

end ratio_transformation_l616_61699


namespace shopkeeper_theft_loss_l616_61619

theorem shopkeeper_theft_loss (cost_price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : 
  profit_rate = 0.1 →
  loss_rate = 0.12 →
  cost_price > 0 →
  let selling_price := cost_price * (1 + profit_rate)
  let loss_value := selling_price * loss_rate
  let loss_percentage := (loss_value / cost_price) * 100
  loss_percentage = 13.2 := by
sorry

end shopkeeper_theft_loss_l616_61619


namespace bowen_total_spent_l616_61650

/-- The price of a pencil in dollars -/
def pencil_price : ℚ := 25/100

/-- The price of a pen in dollars -/
def pen_price : ℚ := 15/100

/-- The number of pens Bowen buys -/
def num_pens : ℕ := 40

/-- The number of pencils Bowen buys -/
def num_pencils : ℕ := num_pens + (2 * num_pens) / 5

/-- The total amount Bowen spends in dollars -/
def total_spent : ℚ := num_pens * pen_price + num_pencils * pencil_price

theorem bowen_total_spent : total_spent = 20 := by sorry

end bowen_total_spent_l616_61650


namespace inscribed_circle_hypotenuse_length_l616_61645

/-- A circle inscribed on the hypotenuse of a right triangle -/
structure InscribedCircle where
  /-- The right triangle -/
  triangle : Set (ℝ × ℝ)
  /-- The inscribed circle -/
  circle : Set (ℝ × ℝ)
  /-- Point A of the triangle -/
  A : ℝ × ℝ
  /-- Point B of the triangle -/
  B : ℝ × ℝ
  /-- Point C of the triangle -/
  C : ℝ × ℝ
  /-- Point M, tangency point of the circle with AB -/
  M : ℝ × ℝ
  /-- Point N, intersection of the circle with AC -/
  N : ℝ × ℝ
  /-- Center of the circle -/
  O : ℝ × ℝ
  /-- The triangle is right-angled at B -/
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  /-- The circle touches AB at M -/
  touches_AB : M ∈ circle ∩ {p | (p.1 - A.1) * (B.2 - A.2) = (p.2 - A.2) * (B.1 - A.1)}
  /-- The circle touches BC -/
  touches_BC : ∃ p ∈ circle, (p.1 - B.1) * (C.2 - B.2) = (p.2 - B.2) * (C.1 - B.1)
  /-- The circle lies on AC -/
  on_AC : O ∈ {p | (p.1 - A.1) * (C.2 - A.2) = (p.2 - A.2) * (C.1 - A.1)}
  /-- AM = 20/9 -/
  AM_length : Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2) = 20/9
  /-- AN:MN = 6:1 -/
  AN_MN_ratio : Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2) / Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) = 6

/-- The main theorem -/
theorem inscribed_circle_hypotenuse_length (ic : InscribedCircle) :
  Real.sqrt ((ic.C.1 - ic.A.1)^2 + (ic.C.2 - ic.A.2)^2) = Real.sqrt 5 + 1/4 := by
  sorry


end inscribed_circle_hypotenuse_length_l616_61645


namespace money_left_l616_61691

def initial_amount : ℕ := 48
def num_books : ℕ := 5
def book_cost : ℕ := 2

theorem money_left : initial_amount - (num_books * book_cost) = 38 := by
  sorry

end money_left_l616_61691


namespace cos_pi_plus_two_alpha_l616_61603

/-- 
Given that the terminal side of angle α passes through point (3,4),
prove that cos(π+2α) = -7/25.
-/
theorem cos_pi_plus_two_alpha (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = 4) → 
  Real.cos (π + 2 * α) = -7/25 := by
sorry

end cos_pi_plus_two_alpha_l616_61603


namespace field_trip_girls_fraction_l616_61629

theorem field_trip_girls_fraction (b : ℚ) (g : ℚ) : 
  g = 2 * b →  -- There are twice as many girls as boys
  (5 / 6 * g) / ((5 / 6 * g) + (1 / 2 * b)) = 10 / 13 := by
  sorry

end field_trip_girls_fraction_l616_61629


namespace run_6000_ends_at_S_S_associated_with_D_or_A_l616_61655

/-- Represents the quarters of the circular track -/
inductive Quarter
| A
| B
| C
| D

/-- Represents a point on the circular track -/
structure Point where
  quarter : Quarter
  distance : ℝ
  h_distance_bound : 0 ≤ distance ∧ distance < 15

/-- The circular track -/
structure Track where
  circumference : ℝ
  h_circumference : circumference = 60

/-- Runner's position after running a given distance -/
def run_position (track : Track) (start : Point) (distance : ℝ) : Point :=
  sorry

/-- Theorem stating that running 6000 feet from point S ends at point S -/
theorem run_6000_ends_at_S (track : Track) (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    run_position track S 6000 = S :=
  sorry

/-- Theorem stating that point S is associated with quarter D or A -/
theorem S_associated_with_D_or_A (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    S.quarter = Quarter.D ∨ S.quarter = Quarter.A :=
  sorry

end run_6000_ends_at_S_S_associated_with_D_or_A_l616_61655


namespace candy_probability_l616_61610

theorem candy_probability : 
  let total_candies : ℕ := 12
  let red_candies : ℕ := 5
  let blue_candies : ℕ := 2
  let green_candies : ℕ := 5
  let pick_count : ℕ := 4
  let favorable_outcomes : ℕ := (red_candies.choose 3) * (blue_candies + green_candies)
  let total_outcomes : ℕ := total_candies.choose pick_count
  (favorable_outcomes : ℚ) / total_outcomes = 14 / 99 := by sorry

end candy_probability_l616_61610


namespace sequence_not_contains_010101_l616_61647

/-- Represents a sequence where each term after the sixth is the last digit of the sum of the previous six terms -/
def Sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 0
  | 2 => 1
  | 3 => 0
  | 4 => 1
  | 5 => 0
  | n + 6 => (Sequence n + Sequence (n + 1) + Sequence (n + 2) + Sequence (n + 3) + Sequence (n + 4) + Sequence (n + 5)) % 10

/-- The weighted sum function used in the proof -/
def S (a b c d e f : ℕ) : ℕ := 2*a + 4*b + 6*c + 8*d + 10*e + 12*f

theorem sequence_not_contains_010101 :
  ∀ n : ℕ, ¬(Sequence n = 0 ∧ Sequence (n + 1) = 1 ∧ Sequence (n + 2) = 0 ∧
            Sequence (n + 3) = 1 ∧ Sequence (n + 4) = 0 ∧ Sequence (n + 5) = 1) :=
by sorry

end sequence_not_contains_010101_l616_61647


namespace min_crossing_time_is_21_l616_61653

/-- Represents a person with their crossing time -/
structure Person where
  name : String
  time : ℕ

/-- Represents the tunnel crossing problem -/
structure TunnelProblem where
  people : List Person
  flashlight : ℕ := 1
  capacity : ℕ := 2

def minCrossingTime (problem : TunnelProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def ourProblem : TunnelProblem :=
  { people := [
      { name := "A", time := 3 },
      { name := "B", time := 4 },
      { name := "C", time := 5 },
      { name := "D", time := 6 }
    ]
  }

theorem min_crossing_time_is_21 :
  minCrossingTime ourProblem = 21 :=
by sorry

end min_crossing_time_is_21_l616_61653


namespace quadratic_form_minimum_l616_61692

theorem quadratic_form_minimum : ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 10 * y ≥ -4.45 := by
  sorry

end quadratic_form_minimum_l616_61692


namespace probability_same_gender_example_l616_61630

/-- Represents a school with a certain number of male and female teachers. -/
structure School :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- Calculates the probability of selecting two teachers of the same gender
    from two given schools. -/
def probability_same_gender (school_a school_b : School) : ℚ :=
  let total_outcomes := (school_a.male_count + school_a.female_count) * (school_b.male_count + school_b.female_count)
  let same_gender_outcomes := school_a.male_count * school_b.male_count + school_a.female_count * school_b.female_count
  same_gender_outcomes / total_outcomes

/-- Theorem stating that the probability of selecting two teachers of the same gender
    from School A (2 males, 1 female) and School B (1 male, 2 females) is 4/9. -/
theorem probability_same_gender_example : 
  probability_same_gender ⟨2, 1⟩ ⟨1, 2⟩ = 4 / 9 := by
  sorry

end probability_same_gender_example_l616_61630


namespace arithmetic_expression_equality_l616_61676

theorem arithmetic_expression_equality : -6 * 5 - (-4 * -2) + (-12 * -6) / 3 = -14 := by
  sorry

end arithmetic_expression_equality_l616_61676


namespace linear_function_passes_through_point_l616_61697

/-- A linear function of the form y = kx + k passes through the point (-1, 0) for any non-zero k. -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ k * x + k
  f (-1) = 0 := by sorry

end linear_function_passes_through_point_l616_61697


namespace trapezoid_angle_bisector_area_ratio_l616_61681

/-- The area ratio of the quadrilateral formed by angle bisector intersections to the trapezoid --/
def area_ratio (a b c d : ℝ) : Set ℝ :=
  {x | x = 1/45 ∨ x = 7/40}

/-- Theorem stating the area ratio for a trapezoid with given side lengths --/
theorem trapezoid_angle_bisector_area_ratio :
  ∀ (a b c d : ℝ),
  a = 5 ∧ b = 15 ∧ c = 15 ∧ d = 20 →
  ∃ (k : ℝ), k ∈ area_ratio a b c d :=
by sorry

end trapezoid_angle_bisector_area_ratio_l616_61681


namespace intersection_of_polar_curves_l616_61639

/-- The intersection point of two polar curves -/
theorem intersection_of_polar_curves (ρ θ : ℝ) :
  ρ ≥ 0 →
  0 ≤ θ →
  θ < π / 2 →
  ρ * Real.cos θ = 3 →
  ρ = 4 * Real.cos θ →
  (ρ = 2 * Real.sqrt 3 ∧ θ = π / 6) :=
by sorry

end intersection_of_polar_curves_l616_61639


namespace rose_orchid_difference_l616_61646

/-- The number of roses initially in the vase -/
def initial_roses : ℕ := 5

/-- The number of orchids initially in the vase -/
def initial_orchids : ℕ := 3

/-- The number of roses finally in the vase -/
def final_roses : ℕ := 12

/-- The number of orchids finally in the vase -/
def final_orchids : ℕ := 2

/-- The difference between the final number of roses and orchids in the vase -/
theorem rose_orchid_difference : final_roses - final_orchids = 10 := by
  sorry

end rose_orchid_difference_l616_61646


namespace no_polyhedron_with_seven_edges_l616_61628

/-- Represents a polyhedron with vertices, edges, and faces. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for polyhedra: V - E + F = 2 -/
axiom eulers_formula (p : Polyhedron) : p.vertices - p.edges + p.faces = 2

/-- Each edge is shared by exactly two faces -/
axiom edge_face_relation (p : Polyhedron) : 2 * p.edges = 3 * p.faces

/-- Theorem: There is no polyhedron with exactly 7 edges -/
theorem no_polyhedron_with_seven_edges :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end no_polyhedron_with_seven_edges_l616_61628


namespace problem_solution_l616_61600

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log x) / (a * x)

theorem problem_solution (a : ℝ) (h_a : a > 0) :
  (∀ x > 0, ∀ y > 0, x < Real.exp 1 → y > Real.exp 1 → f a x < f a y) ∧
  (∀ x > 0, f a x ≤ x - 1/a → a ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → x₂ * Real.log x₁ + x₁ * Real.log x₂ = 0 → x₁ + x₂ > 2) :=
by sorry

end problem_solution_l616_61600


namespace conference_handshakes_count_l616_61624

/-- The number of handshakes at a conference of wizards and elves -/
def conference_handshakes (num_wizards num_elves : ℕ) : ℕ :=
  let wizard_handshakes := num_wizards.choose 2
  let elf_wizard_handshakes := num_wizards * num_elves
  wizard_handshakes + elf_wizard_handshakes

/-- Theorem: The total number of handshakes at the conference is 750 -/
theorem conference_handshakes_count :
  conference_handshakes 25 18 = 750 := by
  sorry

end conference_handshakes_count_l616_61624


namespace arithmetic_expression_equals_24_l616_61670

theorem arithmetic_expression_equals_24 : ∃ (f : List ℝ → ℝ), 
  (f [5, 7, 8, 8] = 24) ∧ 
  (∀ x y z w, f [x, y, z, w] = 
    ((x + y) / z) * w ∨ 
    f [x, y, z, w] = (x - y) * z + w) :=
by sorry

end arithmetic_expression_equals_24_l616_61670


namespace quadratic_inequality_solution_set_l616_61667

theorem quadratic_inequality_solution_set (x : ℝ) : 
  x^2 - x - 6 < 0 ↔ -2 < x ∧ x < 3 := by
  sorry

end quadratic_inequality_solution_set_l616_61667


namespace S_2023_eq_half_l616_61674

def S : ℕ → ℚ
  | 0 => 1 / 2
  | n + 1 => if n % 2 = 0 then 1 / S n else -S n - 1

theorem S_2023_eq_half : S 2022 = 1 / 2 := by sorry

end S_2023_eq_half_l616_61674


namespace plant_branches_l616_61617

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 := by
  sorry

end plant_branches_l616_61617


namespace congruence_problem_l616_61641

theorem congruence_problem (x : ℤ) : 
  (5 * x + 8) % 14 = 3 → (3 * x + 10) % 14 = 7 := by
  sorry

end congruence_problem_l616_61641


namespace calculate_expression_quadratic_equation_roots_l616_61604

-- Problem 1
theorem calculate_expression : 
  (Real.sqrt 2 - Real.sqrt 12 + Real.sqrt (1/2)) * Real.sqrt 3 = 3 * Real.sqrt 6 / 2 - 6 := by sorry

-- Problem 2
theorem quadratic_equation_roots (c : ℝ) (h : (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0) :
  ∃ (x : ℝ), x^2 - 4*x + c = 0 ∧ x ≠ 2 + Real.sqrt 3 ∧ x = 2 - Real.sqrt 3 ∧ c = 1 := by sorry

end calculate_expression_quadratic_equation_roots_l616_61604


namespace second_even_integer_is_78_l616_61690

/-- Given three consecutive even integers where the sum of the first and third is 156,
    prove that the second integer is 78. -/
theorem second_even_integer_is_78 :
  ∀ (a b c : ℤ),
  (b = a + 2) →  -- b is the next consecutive even integer after a
  (c = b + 2) →  -- c is the next consecutive even integer after b
  (a % 2 = 0) →  -- a is even
  (a + c = 156) →  -- sum of first and third is 156
  b = 78 := by
sorry

end second_even_integer_is_78_l616_61690


namespace wonderful_coloring_bounds_l616_61649

/-- A wonderful coloring of a regular polygon is a coloring where no triangle
    formed by its vertices has exactly two colors among its sides. -/
def WonderfulColoring (n : ℕ) (m : ℕ) : Prop := sorry

/-- N is the largest positive integer for which there exists a wonderful coloring
    of a regular N-gon with M colors. -/
def LargestWonderfulN (m : ℕ) : ℕ := sorry

theorem wonderful_coloring_bounds (m : ℕ) (h : m ≥ 3) :
  let n := LargestWonderfulN m
  (n ≤ (m - 1)^2) ∧
  (Nat.Prime (m - 1) → n = (m - 1)^2) := by
  sorry

end wonderful_coloring_bounds_l616_61649


namespace common_rational_root_exists_l616_61608

theorem common_rational_root_exists :
  ∃ (r : ℚ) (a b c d e f g : ℚ),
    (60 * r^4 + a * r^3 + b * r^2 + c * r + 20 = 0) ∧
    (20 * r^5 + d * r^4 + e * r^3 + f * r^2 + g * r + 60 = 0) ∧
    (r > 0) ∧
    (∀ n : ℤ, r ≠ n) ∧
    (r = 1/2) := by
  sorry

end common_rational_root_exists_l616_61608


namespace line_intersects_segment_iff_a_gt_two_l616_61695

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on the positive side of a line -/
def positiveSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c > 0

/-- Check if a point is on the negative side of a line -/
def negativeSide (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c < 0

/-- Check if two points are on opposite sides of a line -/
def oppositeSides (l : Line) (p1 p2 : Point) : Prop :=
  (positiveSide l p1 ∧ negativeSide l p2) ∨ (negativeSide l p1 ∧ positiveSide l p2)

/-- The main theorem -/
theorem line_intersects_segment_iff_a_gt_two (a : ℝ) :
  let A : Point := ⟨1, a⟩
  let B : Point := ⟨2, 4⟩
  let l : Line := ⟨1, -1, 1⟩
  oppositeSides l A B ↔ a > 2 := by
  sorry

end line_intersects_segment_iff_a_gt_two_l616_61695


namespace nine_five_dollar_bills_equal_45_dollars_l616_61685

/-- The total value in dollars when a person has a certain number of five-dollar bills -/
def total_value (num_bills : ℕ) : ℕ := 5 * num_bills

/-- Theorem: If a person has 9 five-dollar bills, they have a total of 45 dollars -/
theorem nine_five_dollar_bills_equal_45_dollars :
  total_value 9 = 45 := by sorry

end nine_five_dollar_bills_equal_45_dollars_l616_61685


namespace complex_equation_solution_l616_61611

theorem complex_equation_solution (z : ℂ) :
  (Complex.I / (z - 1) = (1 : ℂ) / 2) → z = 1 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l616_61611
